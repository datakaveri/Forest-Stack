// web/server/utils/forms/wildlife/animal-rescue.ts
import { and, eq, sql, gte, lte } from 'drizzle-orm';
import { format } from 'date-fns';
import {
  forms,
  formSchemaValues,
  formTemplates,
  dataProducts,
  dataProductSnapshots,
  dataProductSnapshotsData,
} from '../../../db/schema';

/**
 * Interface for a single animal rescue incident
 */
interface AnimalRescueIncident {
  formId: string;
  regionId: string;
  rescueDate: Date; // The actual date of the rescue from Excel
  requestId?: number;
  animalCategory: string;
  animalSpecies: string;
  processStatus: string;
  coordinates: {
    latitude: number;
    longitude: number;
  };
  villageTown?: string;
  beatNaka?: string;
  description?: string;
}

/**
 * Interface for stored rescue incident data
 */
interface StoredRescueIncidentData {
  incidentKey: string;
  formId: string;
  rescueDate: string;
  requestId?: number;
  animalCategory: string;
  animalSpecies: string;
  processStatus: string;
  coordinates: {
    latitude: number;
    longitude: number;
  };
  villageTown?: string;
  beatNaka?: string;
  description?: string;
  recordedAt: string;
}

/**
 * Parse rescue date from DD-MMM-YY or DD-MMM-YYYY format
 * Returns date in UTC to avoid timezone issues
 */
const parseRescueDate = (dateValue: unknown): Date | null => {
  if (!dateValue) return null;

  const dateStr = String(dateValue).trim();
  if (!dateStr) return null;

  // Parse DD-MMM-YY or DD-MMM-YYYY format
  const parts = dateStr.split('-');
  if (parts.length !== 3) return null;

  const [day, month, yearPart] = parts;
  const monthMap: Record<string, number> = {
    Jan: 0,
    Feb: 1,
    Mar: 2,
    Apr: 3,
    May: 4,
    Jun: 5,
    Jul: 6,
    Aug: 7,
    Sep: 8,
    Oct: 9,
    Nov: 10,
    Dec: 11,
  };

  const monthIndex = monthMap[month];
  if (monthIndex === undefined) {
    console.error(`Invalid month in rescue date: ${dateStr}`);
    return null;
  }

  // Handle both 2-digit year (YY) and 4-digit year (YYYY)
  let year: number;
  if (yearPart.length === 2) {
    // Convert 2-digit year to 4-digit year (00-99 → 2000-2099)
    year = 2000 + parseInt(yearPart);
  } else if (yearPart.length === 4) {
    // Use 4-digit year as is
    year = parseInt(yearPart);
  } else {
    console.error(`Invalid year format in rescue date: ${dateStr}`);
    return null;
  }

  // Create date in UTC to avoid timezone issues
  // Using UTC noon (12:00) to avoid any date boundary issues
  return new Date(Date.UTC(year, monthIndex, parseInt(day), 12, 0, 0, 0));
};

/**
 * Extract animal rescue incident data from form
 */
const extractRescueIncident = (
  formId: string,
  regionId: string,
  _reportingDate: Date, // Prefixed with underscore as it's not used
  schemaValues: Array<{ name: string; value: unknown }>,
): AnimalRescueIncident | null => {
  const schemaMap = new Map(schemaValues.map((sv) => [sv.name, sv.value]));

  const latitude = Number(schemaMap.get('latitude'));
  const longitude = Number(schemaMap.get('longitude'));
  const requestId = Number(schemaMap.get('request_id') || 0);
  const animalCategory = String(schemaMap.get('animal_category') || '').trim();
  const animalSpecies = String(schemaMap.get('animal_species') || '').trim();
  const processStatus = String(schemaMap.get('process_status') || '').trim();
  const villageTown = String(schemaMap.get('village_town') || '').trim();
  const beatNaka = String(schemaMap.get('beat_naka') || '').trim();

  // Parse the actual rescue date from the form data (date_of_reporting field)
  const rescueDateValue = schemaMap.get('date_of_reporting');
  const actualRescueDate = parseRescueDate(rescueDateValue);

  if (!actualRescueDate) {
    console.error(
      `Could not parse rescue date for form ${formId}. Raw value: ${rescueDateValue}`,
    );
    return null; // Invalid date means invalid form data
  }

  // Validate required fields
  if (
    Number.isNaN(latitude) ||
    Number.isNaN(longitude) ||
    latitude === 0 ||
    longitude === 0 ||
    !animalCategory ||
    !animalSpecies
  ) {
    return null;
  }

  return {
    formId,
    regionId,
    rescueDate: actualRescueDate, // Uses the actual rescue date from form
    requestId: requestId > 0 ? requestId : undefined,
    animalCategory: animalCategory.toLowerCase(),
    animalSpecies: animalSpecies.toLowerCase(),
    processStatus: processStatus.toLowerCase() || 'unknown',
    coordinates: { latitude, longitude },
    villageTown: villageTown || undefined,
    beatNaka: beatNaka || undefined,
    description: `${animalCategory} - ${animalSpecies} rescue${requestId ? ` (Request: ${requestId})` : ''}`,
  };
};

/**
 * Get or create yearly snapshot for animal rescue
 * Each year gets one snapshot that contains all individual incidents
 */
const getOrCreateYearlySnapshot = async (
  dataProductId: string,
  year: number,
): Promise<string> => {
  const db = useDrizzle();

  // Create UTC dates for the year boundaries
  const startDate = new Date(Date.UTC(year, 0, 1, 0, 0, 0, 0)); // Jan 1, 00:00:00.000 UTC
  const endDate = new Date(Date.UTC(year, 11, 31, 23, 59, 59, 999)); // Dec 31, 23:59:59.999 UTC

  // Format the label with underscores and year only
  const snapshotLabel = `animal_rescue_${year}`;

  console.log(`Creating/fetching yearly snapshot for ${year}:`, {
    startDate: startDate.toISOString(),
    endDate: endDate.toISOString(),
    label: snapshotLabel,
  });

  // Check if snapshot exists
  const snapshot = await db
    .select()
    .from(dataProductSnapshots)
    .where(
      and(
        eq(dataProductSnapshots.dataProductId, dataProductId),
        eq(dataProductSnapshots.label, snapshotLabel),
      ),
    )
    .limit(1);

  // Create if doesn't exist
  if (snapshot.length === 0) {
    const [newSnapshot] = await db
      .insert(dataProductSnapshots)
      .values({
        dataProductId,
        label: snapshotLabel,
        startDate,
        endDate,
        tileFilePath: null,
        createdAt: new Date(),
        updatedAt: new Date(),
        deletedAt: null,
      })
      .returning();

    console.log(`Created new snapshot: ${snapshotLabel} with dates:`, {
      startDate: startDate.toISOString(),
      endDate: endDate.toISOString(),
    });
    return newSnapshot.id;
  }

  return snapshot[0].id;
};

/**
 * Store individual rescue incident as a data point
 */
const storeRescueIncidentDataPoint = async (
  snapshotId: string,
  incident: AnimalRescueIncident,
): Promise<void> => {
  const db = useDrizzle();

  // Create a unique identifier for this incident based on form ID
  const incidentKey = `rescue_${incident.formId}`;

  // Prepare the incident data value
  const incidentData: StoredRescueIncidentData = {
    incidentKey,
    formId: incident.formId,
    rescueDate: incident.rescueDate.toISOString(),
    requestId: incident.requestId,
    animalCategory: incident.animalCategory,
    animalSpecies: incident.animalSpecies,
    processStatus: incident.processStatus,
    coordinates: incident.coordinates,
    villageTown: incident.villageTown,
    beatNaka: incident.beatNaka,
    description: incident.description,
    recordedAt: new Date().toISOString(),
  };

  // Check if this incident already exists (in case of form update)
  const existingData = await db
    .select()
    .from(dataProductSnapshotsData)
    .where(
      and(
        eq(dataProductSnapshotsData.dataProductSnapshotId, snapshotId),
        eq(dataProductSnapshotsData.regionId, incident.regionId),
        sql`(value->>'incidentKey')::text = ${incidentKey}`,
      ),
    )
    .limit(1);

  if (existingData.length > 0) {
    // Update existing incident data
    await db
      .update(dataProductSnapshotsData)
      .set({
        value: incidentData,
        updatedAt: new Date(),
      })
      .where(eq(dataProductSnapshotsData.id, existingData[0].id));

    console.log(
      `Updated rescue incident data for form ${incident.formId} in region ${incident.regionId}`,
    );
  } else {
    // Insert new incident data
    await db.insert(dataProductSnapshotsData).values({
      dataProductSnapshotId: snapshotId,
      value: incidentData,
      regionId: incident.regionId,
      createdAt: new Date(),
      updatedAt: new Date(),
      deletedAt: null,
    });

    console.log(
      `Created new rescue incident data point for form ${incident.formId} in region ${incident.regionId}`,
    );
  }
};

/**
 * Get aggregated rescue statistics for a region and time period
 */
export const getRescueStatistics = async (
  regionId: string,
  startDate: Date,
  endDate: Date,
): Promise<{
  totalIncidents: number;
  byCategory: Record<string, number>;
  bySpecies: Record<string, number>;
  byStatus: Record<string, number>;
  monthlyBreakdown: Record<string, number>;
} | null> => {
  const db = useDrizzle();

  try {
    // Get the data product
    const rescueDataProduct = await db
      .select()
      .from(dataProducts)
      .where(eq(dataProducts.name, 'animal_rescue'))
      .limit(1);

    if (rescueDataProduct.length === 0) {
      return null;
    }

    // Get relevant snapshots
    const snapshots = await db
      .select()
      .from(dataProductSnapshots)
      .where(
        and(
          eq(dataProductSnapshots.dataProductId, rescueDataProduct[0].id),
          gte(dataProductSnapshots.startDate, startDate),
          lte(dataProductSnapshots.endDate, endDate),
        ),
      );

    if (snapshots.length === 0) {
      return null;
    }

    // Get all rescue incidents for the region and time period
    const incidents = await db
      .select()
      .from(dataProductSnapshotsData)
      .where(
        and(
          sql`data_product_snapshot_id IN (${sql.join(
            snapshots.map((s) => sql`${s.id}`),
            sql`, `,
          )})`,
          eq(dataProductSnapshotsData.regionId, regionId),
          sql`(value->>'rescueDate')::timestamp >= ${startDate}`,
          sql`(value->>'rescueDate')::timestamp <= ${endDate}`,
        ),
      );

    if (incidents.length === 0) {
      return null;
    }

    // Aggregate statistics
    const byCategory: Record<string, number> = {};
    const bySpecies: Record<string, number> = {};
    const byStatus: Record<string, number> = {};
    const monthlyBreakdown: Record<string, number> = {};

    for (const record of incidents) {
      const incident = record.value as StoredRescueIncidentData;

      // By category
      byCategory[incident.animalCategory] =
        (byCategory[incident.animalCategory] || 0) + 1;

      // By species
      bySpecies[incident.animalSpecies] =
        (bySpecies[incident.animalSpecies] || 0) + 1;

      // By status
      byStatus[incident.processStatus] =
        (byStatus[incident.processStatus] || 0) + 1;

      // Monthly breakdown
      const month = format(new Date(incident.rescueDate), 'yyyy-MM');
      monthlyBreakdown[month] = (monthlyBreakdown[month] || 0) + 1;
    }

    return {
      totalIncidents: incidents.length,
      byCategory,
      bySpecies,
      byStatus,
      monthlyBreakdown,
    };
  } catch (error) {
    console.error(
      `Error getting rescue statistics for region ${regionId}:`,
      error,
    );
    return null;
  }
};

/**
 * Main function to update animal rescue data product
 * This now stores individual rescue incidents with daily snapshots
 */
export const updateAnimalRescueDataProduct = async (
  regionId: string,
  startDate: Date,
  endDate: Date,
): Promise<void> => {
  const db = useDrizzle();

  try {
    console.log(
      `🦁 [DEBUG] Starting animal rescue update for region ${regionId}`,
    );

    // Step 1: Get the data product
    const rescueDataProduct = await db
      .select()
      .from(dataProducts)
      .where(eq(dataProducts.name, 'animal_rescue'))
      .limit(1);

    if (rescueDataProduct.length === 0) {
      throw new Error('Animal rescue data product not found');
    }

    const dataProduct = rescueDataProduct[0];

    // Step 2: Get the form template
    const rescueTemplate = await db
      .select({ id: formTemplates.id })
      .from(formTemplates)
      .where(eq(formTemplates.frontendTemplateName, 'animal_rescue'))
      .limit(1);

    if (rescueTemplate.length === 0) {
      console.warn('Animal rescue form template not found');
      return;
    }

    // Step 3: Fetch the specific form that triggered this update
    const formsData = await db
      .select({
        formId: forms.id,
        regionId: forms.formRegionId,
        reportingDate: forms.formReportingDate,
        schemaValue: formSchemaValues,
      })
      .from(forms)
      .leftJoin(formSchemaValues, eq(formSchemaValues.formId, forms.id))
      .where(
        and(
          eq(forms.formTemplateId, rescueTemplate[0].id),
          eq(forms.formRegionId, regionId),
          gte(forms.formReportingDate, startDate),
          lte(forms.formReportingDate, endDate),
          sql`${forms.status} IN ('submitted', 'acknowledged')`,
        ),
      );

    if (formsData.length === 0) {
      console.log(`No rescue forms to process for region ${regionId}`);
      return;
    }

    // Step 4: Group schema values by form
    const formsGrouped = new Map<
      string,
      {
        formId: string;
        regionId: string;
        reportingDate: Date;
        schemaValues: Array<{ name: string; value: unknown }>;
      }
    >();

    for (const row of formsData) {
      if (!row.formId || !row.regionId || !row.reportingDate) continue;

      if (!formsGrouped.has(row.formId)) {
        formsGrouped.set(row.formId, {
          formId: row.formId,
          regionId: row.regionId,
          reportingDate: row.reportingDate,
          schemaValues: [],
        });
      }

      if (row.schemaValue) {
        formsGrouped.get(row.formId)!.schemaValues.push({
          name: row.schemaValue.name,
          value: row.schemaValue.value,
        });
      }
    }

    console.log(
      `🦁 [DEBUG] Processing ${formsGrouped.size} rescue forms for individual storage`,
    );

    // Step 5: Process each form as an individual rescue incident
    for (const [formId, formData] of formsGrouped) {
      const incident = extractRescueIncident(
        formId,
        formData.regionId,
        formData.reportingDate,
        formData.schemaValues,
      );

      if (!incident) {
        console.warn(
          `Could not extract valid rescue incident data from form ${formId}`,
        );
        continue;
      }

      // Get or create the yearly snapshot for the rescue year
      const year = incident.rescueDate.getUTCFullYear();
      const snapshotId = await getOrCreateYearlySnapshot(dataProduct.id, year);

      // Store the individual rescue incident
      await storeRescueIncidentDataPoint(snapshotId, incident);
    }

    console.log(
      `✅ Successfully processed ${formsGrouped.size} rescue incidents for region ${regionId}`,
    );

    // Step 6: Optionally calculate and log statistics
    const stats = await getRescueStatistics(regionId, startDate, endDate);
    if (stats) {
      console.log(`Animal rescue statistics for region ${regionId}:`, {
        totalIncidents: stats.totalIncidents,
        categories: Object.keys(stats.byCategory).length,
        species: Object.keys(stats.bySpecies).length,
      });
    }
  } catch (error) {
    console.error(
      `Error updating animal rescue data product for region ${regionId}:`,
      error,
    );
    throw error;
  }
};

/**
 * Delete a rescue incident data point when a form is deleted
 */
export const deleteRescueIncident = async (
  formId: string,
  regionId: string,
  _reportingDate: Date, // Prefixed with underscore as it's not used
): Promise<void> => {
  const db = useDrizzle();

  try {
    // Get the data product
    const rescueDataProduct = await db
      .select()
      .from(dataProducts)
      .where(eq(dataProducts.name, 'animal_rescue'))
      .limit(1);

    if (rescueDataProduct.length === 0) {
      return;
    }

    // Need to find the actual rescue date from the form to locate the snapshot
    const formData = await db
      .select({
        name: formSchemaValues.name,
        value: formSchemaValues.value,
      })
      .from(formSchemaValues)
      .where(
        and(
          eq(formSchemaValues.formId, formId),
          eq(formSchemaValues.name, 'date_of_reporting'),
        ),
      )
      .limit(1);

    if (formData.length === 0) {
      return;
    }

    const rescueDate = parseRescueDate(formData[0].value);
    if (!rescueDate) {
      return;
    }

    // Get the snapshot for the year
    const year = rescueDate.getUTCFullYear();
    const snapshotLabel = `animal_rescue_${year}`;

    const snapshot = await db
      .select()
      .from(dataProductSnapshots)
      .where(
        and(
          eq(dataProductSnapshots.dataProductId, rescueDataProduct[0].id),
          eq(dataProductSnapshots.label, snapshotLabel),
        ),
      )
      .limit(1);

    if (snapshot.length === 0) {
      return;
    }

    // Delete the rescue incident data point
    const incidentKey = `rescue_${formId}`;

    await db
      .update(dataProductSnapshotsData)
      .set({
        deletedAt: new Date(),
        updatedAt: new Date(),
      })
      .where(
        and(
          eq(dataProductSnapshotsData.dataProductSnapshotId, snapshot[0].id),
          eq(dataProductSnapshotsData.regionId, regionId),
          sql`(value->>'incidentKey')::text = ${incidentKey}`,
        ),
      );

    console.log(`Deleted rescue incident data for form ${formId}`);
  } catch (error) {
    console.error(`Error deleting rescue incident for form ${formId}:`, error);
    // Don't throw - deletion failures shouldn't break other operations
  }
};
