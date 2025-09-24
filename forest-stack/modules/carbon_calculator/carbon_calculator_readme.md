# Carbon Calculator – Detailed Documentation

This module provides comprehensive carbon sequestration calculations for forest management and carbon credit projects. It calculates CO2 storage, biomass accumulation, and carbon sequestration rates for multiple tree species over time, supporting forest carbon accounting and climate change mitigation planning.

---

## 1. What the module does

### Core Functionality

1. **Carbon Sequestration Calculation**:
   - Calculates CO2 storage per tree based on species-specific growth parameters
   - Handles multiple survival rates (40%, 60%, 80%, 100%)
   - Supports harvest cycles and carbon retention post-harvest
   - Provides both annual and cumulative carbon calculations

2. **Biomass Modeling**:
   - Above-ground biomass (AGB) calculation using volume formulas
   - Below-ground biomass (BGB) using root-shoot ratios
   - Stem, leaf, and branch biomass using biomass expansion factors
   - Total biomass calculation for comprehensive carbon accounting

3. **Multi-Species Support**:
   - Species-specific growth parameters and formulas
   - Individual species tracking and analysis
   - Mixed-species plantation calculations
   - Flexible species configuration management

4. **Temporal Analysis**:
   - Monthly growth increments and diameter calculations
   - Annual carbon sequestration projections
   - Long-term carbon storage modeling (up to 30 years)
   - Harvest cycle management and carbon retention

---

## 2. Data Models and Interfaces

### Core Data Types

#### `SpeciesData`
```typescript
interface SpeciesData {
  name: string;                           // Species name
  formula: string;                        // Volume calculation formula
  averageIncrementInDiameter: string;     // Annual diameter increment (cm)
  woodDensity: string;                    // Wood density (g/cm³)
  biomassExpansionFactor: string;         // BEF for stem/leaf/branch biomass
  carbonFraction: string;                 // Carbon content fraction
  co2ConversionFactor: string;            // CO2 conversion factor (default 3.67)
  rootShootRatio: string;                 // Root to shoot ratio
}
```

#### `SpeciesConfig`
```typescript
interface SpeciesConfig {
  speciesData: SpeciesData;               // Species parameters
  numTrees: number;                       // Number of trees
  harvestCycle?: number;                  // Harvest cycle in years
  carbonRetentionPostHarvest?: number;    // Carbon retention % after harvest
  plantationDate?: string;                // Planting date (MM-YYYY format)
}
```

#### `CO2CalculationResult`
```typescript
interface CO2CalculationResult {
  totalCO2Stock: number;                  // Total CO2 stored (kg)
  annualSequestration: number;            // Annual CO2 sequestration (kg)
  isHarvestYear: boolean;                 // Whether this is a harvest year
}
```

---

## 3. Core Calculation Functions

### `volumeFromDiameter(diameterCm: number, formula: string): number`
Calculates tree volume from diameter using species-specific formulas.

**Parameters**:
- `diameterCm`: Tree diameter in centimeters
- `formula`: Mathematical formula string (e.g., "0.0001 * D^2.5")

**Returns**: Volume in cubic meters

**Example**:
```typescript
const volume = volumeFromDiameter(25.5, "0.0001 * D^2.5");
// Returns volume for 25.5cm diameter tree
```

### `co2StockPerTree(months: number, speciesData: SpeciesData, survivalRate: number): number`
Calculates CO2 stock per tree at given months of growth.

**Parameters**:
- `months`: Number of months since planting
- `speciesData`: Species-specific parameters
- `survivalRate`: Tree survival rate (40-100%)

**Process**:
1. Calculates diameter from monthly increments
2. Computes volume using species formula
3. Calculates biomass components (AGB, BGB, SLB)
4. Converts to carbon and CO2 equivalents

### `calculateCO2PerHectare(speciesData, numTrees, areaHectare, year, survivalRate): number`
Calculates CO2 per hectare for a specific species at a given year.

**Parameters**:
- `speciesData`: Species parameters
- `numTrees`: Number of trees
- `areaHectare`: Area in hectares
- `year`: Years since planting
- `survivalRate`: Survival rate percentage

**Returns**: CO2 per hectare in metric tons

---

## 4. Advanced Calculation Functions

### `calculateCarbonSequestration(speciesDataArray, survivalRates): CalculationResult`
Main function for comprehensive carbon sequestration analysis.

**Parameters**:
- `speciesDataArray`: Array of species configurations
- `survivalRates`: Array of survival rates to analyze [40, 60, 80, 100]

**Returns**:
- Annual carbon sequestration data
- Cumulative carbon storage
- Chart data for visualization
- Period-based analysis

### `calculateCumulativeCO2(speciesDataArray, years?, survivalRates): CumulativeCO2Result`
Calculates cumulative CO2 storage over time with chart data.

**Features**:
- Multi-species analysis
- Survival rate comparisons
- Harvest cycle management
- Excel-compatible output format

---

## 5. Carbon Calculator Class

### `CarbonCalculator` Class
Main class for managing species configurations and calculations.

#### Constructor
```typescript
const calculator = new CarbonCalculator(speciesConfigs);
```

#### Key Methods

##### `addSpecies(speciesConfig: SpeciesConfig): void`
Adds a new species configuration to the calculator.

##### `removeSpecies(speciesName: string): void`
Removes a species configuration by name.

##### `calculateSequestration(survivalRates?: number[]): CalculationResult`
Calculates carbon sequestration for all configured species.

##### `calculateCumulativeCO2(years?: number[], survivalRates?: number[]): CumulativeCO2Result`
Calculates cumulative CO2 with optional year filtering.

##### `calculateSpeciesCO2(speciesName: string, year: number, survivalRate: number): number`
Calculates CO2 for a specific species at a given year.

##### `calculateSpeciesCO2PerHectare(speciesName: string, areaHectare: number, year: number, survivalRate: number): number`
Calculates CO2 per hectare for a specific species.

---

## 6. Usage Examples

### Basic Species Configuration
```typescript
import { CarbonCalculator, SpeciesConfig } from './carbon_calculator';

const speciesData = {
  name: "Eucalyptus",
  formula: "0.0001 * D^2.5",
  averageIncrementInDiameter: "2.5",
  woodDensity: "0.6",
  biomassExpansionFactor: "1.3",
  carbonFraction: "0.47",
  co2ConversionFactor: "3.67",
  rootShootRatio: "0.25"
};

const speciesConfig: SpeciesConfig = {
  speciesData,
  numTrees: 1000,
  harvestCycle: 7,
  carbonRetentionPostHarvest: 20,
  plantationDate: "06-2023"
};

const calculator = new CarbonCalculator([speciesConfig]);
```

### Calculate Carbon Sequestration
```typescript
// Calculate for all survival rates
const result = calculator.calculateSequestration([40, 60, 80, 100]);

// Get annual data
console.log(result.annualData);
// Output: { "Jun 2023-Dec 2023": { "Eucalyptus": 15.2, "Total": 15.2 } }

// Get cumulative data
console.log(result.cumulativeData);
// Output: { "Jun 2023-Dec 2023": { "Eucalyptus": 15.2, "Total": 15.2 } }
```

### Calculate CO2 Per Hectare
```typescript
const co2PerHectare = calculator.calculateSpeciesCO2PerHectare(
  "Eucalyptus", 
  5.0,  // 5 hectares
  5,    // 5 years
  80    // 80% survival rate
);
console.log(`CO2 per hectare: ${co2PerHectare} metric tons`);
```

### Multi-Species Analysis
```typescript
const mixedSpecies = [
  {
    speciesData: eucalyptusData,
    numTrees: 500,
    plantationDate: "06-2023"
  },
  {
    speciesData: teakData,
    numTrees: 300,
    plantationDate: "06-2023"
  }
];

const calculator = new CarbonCalculator(mixedSpecies);
const result = calculator.calculateCumulativeCO2();
```

---

## 7. Biomass Calculation Methodology

### Above-Ground Biomass (AGB)
```
AGB = Volume × Wood Density
```

### Below-Ground Biomass (BGB)
```
BGB = AGB × Root-Shoot Ratio
```

### Stem, Leaf, and Branch Biomass (SLB)
```
SLB = AGB × Biomass Expansion Factor
```

### Total Biomass (TBM)
```
TBM = AGB + BGB + SLB
```

### Carbon Content
```
Carbon = TBM × Carbon Fraction
Live Carbon = Carbon × (Survival Rate / 100)
```

### CO2 Equivalent
```
CO2 = Live Carbon × CO2 Conversion Factor
```

---

## 8. Harvest Cycle Management

### Harvest Logic
- Harvest occurs at rotation anniversary
- Carbon retention percentage applied post-harvest
- Calculation stops after harvest (unless replanting)

### Carbon Retention
- Configurable percentage of carbon retained after harvest
- Accounts for long-lived wood products
- Supports carbon credit calculations

---

## 9. Dependencies

### Core Dependencies
- `date-fns` - Date manipulation and calculations
- TypeScript - Type safety and development experience

### Optional Dependencies
- `@types/node` - Node.js type definitions
- `ts-node` - TypeScript execution

---

## 10. Configuration

### Species Parameters
Each species requires specific parameters for accurate calculations:

- **Growth Rate**: Annual diameter increment
- **Wood Density**: Species-specific density values
- **Biomass Factors**: Expansion factors for different biomass components
- **Carbon Content**: Fraction of biomass that is carbon
- **Root-Shoot Ratio**: Below-ground to above-ground biomass ratio

### Default Values
- **CO2 Conversion Factor**: 3.67 (standard IPCC value)
- **Survival Rates**: [40, 60, 80, 100]%
- **Analysis Period**: Up to 30 years

---

## 11. Output Data Structure

### Annual Data
```typescript
{
  "Jun 2023-Dec 2023": {
    "Eucalyptus": 15.2,
    "Teak": 8.7,
    "Total": 23.9
  },
  "Jan 2024-Dec 2024": {
    "Eucalyptus": 45.6,
    "Teak": 26.1,
    "Total": 71.7
  }
}
```

### Chart Data
```typescript
[
  {
    year: 1,
    period: "Jun 2023-Dec 2023",
    survival40: 15200000,  // CO2 in kg
    survival60: 22800000,
    survival80: 30400000,
    survival100: 38000000
  }
]
```

---

## 12. Integration with Forest-Stack

### Forest Management Applications
- **Carbon Credit Projects**: Quantify carbon sequestration for carbon markets
- **Forest Planning**: Optimize species selection for carbon benefits
- **Climate Mitigation**: Assess forest contribution to climate goals
- **Financial Analysis**: Calculate carbon value for forest investments

### Data Products
- **Carbon Inventories**: Annual carbon stock assessments
- **Sequestration Reports**: Projected carbon capture over time
- **Species Analysis**: Individual species carbon contributions
- **Harvest Planning**: Carbon retention and harvest optimization

---

## 13. Performance Considerations

### Calculation Efficiency
- **Vectorized Operations**: Efficient array processing
- **Caching**: Species data caching for repeated calculations
- **Memory Management**: Optimized data structures for large datasets

### Scalability
- **Multi-Species**: Handles hundreds of species configurations
- **Long-term Analysis**: Efficient 30-year projections
- **Batch Processing**: Supports multiple plantation scenarios

---

## 14. Troubleshooting

### Common Issues

1. **Formula Errors**: Invalid mathematical expressions in species formulas
2. **Date Format**: Incorrect plantation date format (use MM-YYYY)
3. **Parameter Validation**: Missing or invalid species parameters
4. **Memory Issues**: Large datasets causing performance problems

### Debug Tips
- Validate species formulas before calculation
- Check plantation date formats
- Verify all required parameters are provided
- Test with small datasets first

### Error Handling
- Formula evaluation errors are caught and logged
- Missing species configurations throw descriptive errors
- Invalid dates are handled gracefully

---

## 15. Future Enhancements

### Potential Improvements
- **Climate Factors**: Temperature and precipitation effects on growth
- **Soil Conditions**: Soil type impact on carbon sequestration
- **Pest/Disease**: Mortality factors in survival calculations
- **Management Practices**: Thinning and pruning effects

### Integration Opportunities
- **GIS Integration**: Spatial analysis of carbon sequestration
- **Real-time Data**: Integration with forest monitoring systems
- **API Development**: RESTful API for carbon calculations
- **Visualization**: Interactive charts and maps
