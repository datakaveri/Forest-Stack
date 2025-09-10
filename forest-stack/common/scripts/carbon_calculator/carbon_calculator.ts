// carbon-calculator.ts
import {
    addMonths,
    startOfMonth,
    endOfYear,
    addYears,
    differenceInMonths,
    format,
  } from 'date-fns';
  
  // ============================================================================
  // TYPES
  // ============================================================================
  
  export interface SpeciesData {
    name: string;
    formula: string;
    averageIncrementInDiameter: string;
    woodDensity: string;
    biomassExpansionFactor: string;
    carbonFraction: string;
    co2ConversionFactor: string;
    rootShootRatio: string;
  }
  
  export interface SpeciesConfig {
    speciesData: SpeciesData;
    numTrees: number;
    harvestCycle?: number;
    carbonRetentionPostHarvest?: number;
    plantationDate?: string;
  }
  
  export interface SpeciesCarbonData {
    speciesData: SpeciesData;
    numTrees: number;
    areaHectare: number;
    year: 1 | 3 | 5 | 7 | 10 | 15 | 20 | 25 | 30;
    survivalRate: 40 | 60 | 80 | 100;
  }
  
  export interface CO2CalculationResult {
    totalCO2Stock: number;
    annualSequestration: number;
    isHarvestYear: boolean;
  }
  
  export interface CumulativeCO2Result {
    chartData: Array<{
      year: number;
      period: string;
      survival40?: number;
      survival60?: number;
      survival80?: number;
      survival100?: number;
    }>;
    carbonSequestrationForCurrentYear: number;
    speciesBreakdown: Array<{
      speciesIndex: number;
      yearlyData: Array<{
        year: number;
        totalCO2: number;
        isHarvestYear: boolean;
      }>;
    }>;
    annualData: Record<string, Record<string, number>>;
    cumulativeData: Record<string, Record<string, number>>;
    periods: string[];
  }
  
  interface CalculationPeriod {
    start: Date;
    end: Date;
    label: string;
  }
  
  interface PeriodDelta {
    period: string;
    species: string;
    deltaMillion: number;
    order: Date;
    survivalRate: number;
  }
  
  interface ChartDataPoint {
    year: number;
    period: string;
    survival40?: number;
    survival60?: number;
    survival80?: number;
    survival100?: number;
  }
  
  interface CalculationResult {
    annualData: Record<string, Record<string, number>>;
    cumulativeData: Record<string, Record<string, number>>;
    periods: string[];
    chartData: ChartDataPoint[];
  }
  
  // ============================================================================
  // CORE CALCULATION FUNCTIONS
  // ============================================================================
  
  /**
   * Calculate volume from diameter using formula string
   */
  const volumeFromDiameter = (diameterCm: number, formula: string): number => {
    if (diameterCm <= 0) return 0;
  
    try {
      // Create a function from the formula string
      // The formula expects 'D' as a variable
      const volumeFunction = new Function('D', 'Math', `return ${formula}`);
      return volumeFunction(diameterCm, Math);
    } catch (e) {
      console.error('Error evaluating formula:', formula, e);
      return 0;
    }
  };
  
  /**
   * Calculate CO2 stock per tree at given months of growth
   */
  const co2StockPerTree = (
    months: number,
    speciesData: SpeciesData,
    survivalRate: number,
  ): number => {
    // Parse numeric values
    const avgDiameterIncrement = parseFloat(
      speciesData.averageIncrementInDiameter || '0',
    );
    const woodDensity = parseFloat(speciesData.woodDensity);
    const rsRatio = parseFloat(speciesData.rootShootRatio);
    const bef = parseFloat(speciesData.biomassExpansionFactor);
    const carbonFraction = parseFloat(speciesData.carbonFraction);
    const co2Factor = parseFloat(speciesData.co2ConversionFactor || '3.67');
  
    // Calculate diameter after given months
    const diameter = avgDiameterIncrement * (months / 12.0);
  
    const volume = volumeFromDiameter(diameter, speciesData.formula);
  
    // Biomass calculations
    const agb = volume * woodDensity;
    const bgb = agb * rsRatio;
    const slb = agb * bef;
    const tbm = agb + bgb + slb;
  
    // Carbon calculations
    const carbon = tbm * carbonFraction;
    const liveCarbon = carbon * (survivalRate / 100); // Convert survival rate from percentage
  
    return liveCarbon * co2Factor;
  };
  
  /**
   * Build calculation periods following Python implementation
   */
  const buildPeriods = (
    earliestPlantDate: Date,
    horizonYears: number = 30,
  ): CalculationPeriod[] => {
    const periods: CalculationPeriod[] = [];
  
    // First partial slice
    const firstStart = startOfMonth(addMonths(earliestPlantDate, 1));
    const firstEnd = addYears(endOfYear(firstStart), 0);
    firstEnd.setDate(firstEnd.getDate() + 1); // First day of next year
  
    periods.push({
      start: firstStart,
      end: firstEnd,
      label: `${format(firstStart, 'MMM yyyy')}-${format(new Date(firstEnd.getTime() - 86400000), 'MMM yyyy')}`,
    });
  
    // Full-year slices
    const horizon = addYears(earliestPlantDate, horizonYears);
    let cursor = firstEnd;
  
    while (cursor < horizon) {
      const nextEnd = addYears(cursor, 1);
  
      if (nextEnd > horizon) {
        // Final partial slice
        const monthAfter = (earliestPlantDate.getMonth() % 12) + 1;
        const yearAfter =
          horizon.getFullYear() + (earliestPlantDate.getMonth() === 11 ? 1 : 0);
        const finalEnd = new Date(yearAfter, monthAfter, 1);
  
        periods.push({
          start: cursor,
          end: finalEnd,
          label: `${format(cursor, 'MMM yyyy')}-${format(new Date(finalEnd.getTime() - 86400000), 'MMM yyyy')}`,
        });
        break;
      }
  
      periods.push({
        start: cursor,
        end: nextEnd,
        label: `${format(cursor, 'MMM yyyy')}-${format(new Date(nextEnd.getTime() - 86400000), 'MMM yyyy')}`,
      });
  
      cursor = nextEnd;
    }
  
    return periods;
  };
  
  /**
   * Calculate carbon sequestration for a specific survival rate
   */
  const calculateForSurvivalRate = (
    speciesDataArray: SpeciesConfig[],
    periods: CalculationPeriod[],
    survivalRate: number,
  ): PeriodDelta[] => {
    const records: PeriodDelta[] = [];
  
    speciesDataArray.forEach((speciesConfig) => {
      const plantDateParts = speciesConfig.plantationDate!.split('-');
      const plantMonth = parseInt(plantDateParts[0]) - 1;
      const plantYear = parseInt(plantDateParts[1]);
  
      // Set to last day of plantation month
      const plantDate = new Date(plantYear, plantMonth + 1, 0);
      const plantDatePlusOne = new Date(plantDate);
      plantDatePlusOne.setDate(plantDatePlusOne.getDate() + 1);
  
      let cumulativeMonths = 0;
      let previousStock = 0;
  
      const harvestCycle = speciesConfig.harvestCycle || 0;
      const retention = speciesConfig.carbonRetentionPostHarvest
        ? speciesConfig.carbonRetentionPostHarvest / 100
        : 0;
  
      for (const period of periods) {
        // Skip periods that end before planting
        if (period.end <= plantDatePlusOne) continue;
  
        // Adjust slice start for this event
        const sliceStart =
          period.start > plantDatePlusOne ? period.start : plantDatePlusOne;
        const sliceEnd = period.end;
  
        // Calculate months in slice
        const monthsInSlice = differenceInMonths(sliceEnd, sliceStart);
        const monthsPrev = cumulativeMonths;
        cumulativeMonths += monthsInSlice;
        const monthsEnd = cumulativeMonths;
  
        // Calculate stock at end of period
        let stock =
          co2StockPerTree(monthsEnd, speciesConfig.speciesData, survivalRate) *
          speciesConfig.numTrees;
  
        // Check for harvest
        if (harvestCycle > 0) {
          const yearsPrev = monthsPrev / 12.0;
          const yearsEnd = monthsEnd / 12.0;
  
          if (
            Math.floor(yearsPrev / harvestCycle) <
            Math.floor(yearsEnd / harvestCycle)
          ) {
            // Harvest at rotation anniversary
            stock =
              co2StockPerTree(
                harvestCycle * 12,
                speciesConfig.speciesData,
                survivalRate,
              ) *
              speciesConfig.numTrees *
              retention;
  
            const delta = stock - previousStock;
            records.push({
              period: period.label,
              species: speciesConfig.speciesData.name,
              deltaMillion: delta / 1e6,
              order: period.start,
              survivalRate,
            });
  
            previousStock = stock;
            break; // Stop after harvest
          }
        }
  
        // Normal period calculation
        const delta = stock - previousStock;
        records.push({
          period: period.label,
          species: speciesConfig.speciesData.name,
          deltaMillion: delta / 1e6,
          order: period.start,
          survivalRate,
        });
  
        previousStock = stock;
      }
    });
  
    return records;
  };
  
  // ============================================================================
  // PUBLIC API FUNCTIONS
  // ============================================================================
  
  /**
   * Calculate CO2 per hectare for a specific species at a given year
   * This function provides backwards compatibility with the old API
   */
  export const calculateCO2PerHectare = (
    speciesData: SpeciesData,
    numTrees: number,
    areaHectare: number,
    year: number,
    survivalRate: number,
  ): number => {
    // Calculate months from years
    const months = year * 12;
    
    // Calculate CO2 per tree
    const co2PerTree = co2StockPerTree(months, speciesData, survivalRate);
    
    // Calculate total CO2 for all trees
    const totalCO2 = co2PerTree * numTrees;
    
    // Convert to metric tons (from kg) and calculate per hectare
    const totalCO2MetricTons = totalCO2 / 1_000;
    
    // Return total CO2 for the given area
    return (totalCO2MetricTons / areaHectare);
  };
  
  /**
   * Calculate CO2 for a specific number of trees at a given year
   * This provides the raw CO2 value without area normalization
   */
  export const calculateCO2 = (
    speciesData: SpeciesData,
    numTrees: number,
    year: number,
    survivalRate: number,
  ): number => {
    // Calculate months from years
    const months = year * 12;
    
    // Calculate CO2 per tree
    const co2PerTree = co2StockPerTree(months, speciesData, survivalRate);
    
    // Calculate total CO2 for all trees (returns in kg)
    return co2PerTree * numTrees;
  };
  
  /**
   * Calculate carbon sequestration following Python implementation
   */
  export const calculateCarbonSequestration = (
    speciesDataArray: SpeciesConfig[],
    survivalRates: number[] = [40, 60, 80, 100],
  ): CalculationResult & { chartData: ChartDataPoint[] } => {
    // Find earliest plantation date
    const plantationDates = speciesDataArray
      .filter((config) => config.plantationDate)
      .map((config) => {
        const [month, year] = config.plantationDate!.split('-');
        return new Date(parseInt(year), parseInt(month) - 1, 1);
      });
  
    if (plantationDates.length === 0) {
      throw new Error('No plantation dates found');
    }
  
    const earliestDate = new Date(
      Math.min(...plantationDates.map((d) => d.getTime())),
    );
  
    // Set earliest date to last day of month
    earliestDate.setMonth(earliestDate.getMonth() + 1);
    earliestDate.setDate(0);
  
    // Build periods
    const periods = buildPeriods(earliestDate);
  
    // Calculate for default survival rate (80% as per Python, but we'll use 60% for the main calculation)
    const mainSurvivalRate = 60;
    const records = calculateForSurvivalRate(
      speciesDataArray,
      periods,
      mainSurvivalRate,
    );
  
    // Build annual and cumulative tables
    const periodOrder = [...new Set(records.map((r) => r.period))];
    const species = [...new Set(records.map((r) => r.species))];
  
    // Initialize data structures
    const annualData: Record<string, Record<string, number>> = {};
    const cumulativeData: Record<string, Record<string, number>> = {};
  
    periodOrder.forEach((period) => {
      annualData[period] = {};
      cumulativeData[period] = {};
  
      let periodTotal = 0;
      species.forEach((sp) => {
        const value = records
          .filter((r) => r.period === period && r.species === sp)
          .reduce((sum, r) => sum + r.deltaMillion, 0);
  
        annualData[period][sp] = value;
        periodTotal += value;
      });
  
      annualData[period]['Total'] = periodTotal;
    });
  
    // Calculate cumulative values
    const cumulativeTotals: Record<string, number> = {};
    species.forEach((sp) => (cumulativeTotals[sp] = 0));
    cumulativeTotals['Total'] = 0;
  
    periodOrder.forEach((period) => {
      species.forEach((sp) => {
        cumulativeTotals[sp] += annualData[period][sp] || 0;
        cumulativeData[period][sp] = cumulativeTotals[sp];
      });
      cumulativeTotals['Total'] += annualData[period]['Total'] || 0;
      cumulativeData[period]['Total'] = cumulativeTotals['Total'];
    });
  
    // Calculate chart data for all survival rates
    const chartData: ChartDataPoint[] = [];
    if (survivalRates.length > 0) {
      periodOrder.forEach((period, index) => {
        const dataPoint: ChartDataPoint = {
          year: index + 1,
          period,
        };
  
        survivalRates.forEach((rate) => {
          const rateRecords = calculateForSurvivalRate(
            speciesDataArray,
            periods,
            rate,
          );
          const rateAnnual: Record<string, number> = {};
  
          rateRecords
            .filter((r) => r.period === period)
            .forEach((r) => {
              rateAnnual[r.species] =
                (rateAnnual[r.species] || 0) + r.deltaMillion;
            });
  
          // Calculate cumulative up to this period
          let cumulative = 0;
          for (let i = 0; i <= index; i++) {
            const periodRecords = rateRecords.filter(
              (r) => r.period === periodOrder[i],
            );
            cumulative += periodRecords.reduce(
              (sum, r) => sum + r.deltaMillion,
              0,
            );
          }
  
          const survivalKey = `survival${rate}` as keyof ChartDataPoint;
          if (
            survivalKey === 'survival40' ||
            survivalKey === 'survival60' ||
            survivalKey === 'survival80' ||
            survivalKey === 'survival100'
          ) {
            dataPoint[survivalKey] = cumulative * 1e6; // Convert back to tonnes for chart
          }
        });
  
        chartData.push(dataPoint);
      });
    }
  
    return {
      annualData,
      cumulativeData,
      periods: periodOrder,
      chartData,
    };
  };
  
  /**
   * Convert to chart format for backwards compatibility
   */
  export const calculateCumulativeCO2 = (
    speciesDataArray: SpeciesConfig[],
    years?: number[],
    survivalRates: number[] = [40, 60, 80, 100],
  ): CumulativeCO2Result => {
    const result = calculateCarbonSequestration(speciesDataArray, survivalRates);
  
    // Filter chart data to match requested years if provided
    let chartData = result.chartData;
    if (years && years.length > 0) {
      chartData = years.map((year) => {
        // Find the closest period for the requested year
        const targetIndex = Math.min(year - 1, result.chartData.length - 1);
        if (targetIndex >= 0 && result.chartData[targetIndex]) {
          return result.chartData[targetIndex];
        }
        return {
          year,
          period: '',
          survival40: 0,
          survival60: 0,
          survival80: 0,
          survival100: 0,
        };
      });
    }
  
    // Get current year sequestration
    const lastPeriod = result.periods[result.periods.length - 1];
    const carbonSequestrationForCurrentYear =
      result.annualData[lastPeriod]?.['Total'] || 0;
  
    // Species breakdown
    const species = [...new Set(speciesDataArray.map((s) => s.speciesData.name))];
    const speciesBreakdown = species.map((speciesName, index) => ({
      speciesIndex: index,
      yearlyData: result.periods.map((period, yearIndex) => ({
        year: yearIndex + 1,
        totalCO2: result.annualData[period][speciesName] || 0,
        isHarvestYear: false, // This would need to be tracked in the main calculation
      })),
    }));
  
    return {
      chartData,
      carbonSequestrationForCurrentYear,
      speciesBreakdown,
      // Additional data for the Excel export
      annualData: result.annualData,
      cumulativeData: result.cumulativeData,
      periods: result.periods,
    };
  };
  
  // ============================================================================
  // CARBON CALCULATOR CLASS
  // ============================================================================
  
  /**
   * Main Carbon Calculator class for easy usage
   */
  export class CarbonCalculator {
    private speciesConfigs: SpeciesConfig[];
  
    constructor(speciesConfigs: SpeciesConfig[]) {
      this.speciesConfigs = speciesConfigs;
    }
  
    /**
     * Add a species configuration
     */
    addSpecies(speciesConfig: SpeciesConfig): void {
      this.speciesConfigs.push(speciesConfig);
    }
  
    /**
     * Remove a species configuration by name
     */
    removeSpecies(speciesName: string): void {
      this.speciesConfigs = this.speciesConfigs.filter(
        config => config.speciesData.name !== speciesName
      );
    }
  
    /**
     * Get all species configurations
     */
    getSpecies(): SpeciesConfig[] {
      return [...this.speciesConfigs];
    }
  
    /**
     * Calculate carbon sequestration for all species
     */
    calculateSequestration(survivalRates: number[] = [40, 60, 80, 100]) {
      return calculateCarbonSequestration(this.speciesConfigs, survivalRates);
    }
  
    /**
     * Calculate cumulative CO2 for all species
     */
    calculateCumulativeCO2(years?: number[], survivalRates: number[] = [40, 60, 80, 100]) {
      return calculateCumulativeCO2(this.speciesConfigs, years, survivalRates);
    }
  
    /**
     * Calculate CO2 for a specific species
     */
    calculateSpeciesCO2(speciesName: string, year: number, survivalRate: number): number {
      const speciesConfig = this.speciesConfigs.find(
        config => config.speciesData.name === speciesName
      );
      
      if (!speciesConfig) {
        throw new Error(`Species ${speciesName} not found`);
      }
  
      return calculateCO2(speciesConfig.speciesData, speciesConfig.numTrees, year, survivalRate);
    }
  
    /**
     * Calculate CO2 per hectare for a specific species
     */
    calculateSpeciesCO2PerHectare(speciesName: string, areaHectare: number, year: number, survivalRate: number): number {
      const speciesConfig = this.speciesConfigs.find(
        config => config.speciesData.name === speciesName
      );
      
      if (!speciesConfig) {
        throw new Error(`Species ${speciesName} not found`);
      }
  
      return calculateCO2PerHectare(speciesConfig.speciesData, speciesConfig.numTrees, areaHectare, year, survivalRate);
    }
  }
  