from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

AGE_BUCKET_OPTIONS = (
    "[0-10)",
    "[10-20)",
    "[20-30)",
    "[30-40)",
    "[40-50)",
    "[50-60)",
    "[60-70)",
    "[70-80)",
    "[80-90)",
    "[90-100)",
)
WEIGHT_BUCKET_OPTIONS = (
    "[0-25)",
    "[25-50)",
    "[50-75)",
    "[75-100)",
    "[100-125)",
    "[125-150)",
    "[150-175)",
    "[175-200)",
    ">200",
)
RACE_OPTIONS = ("AfricanAmerican", "Asian", "Caucasian", "Hispanic", "Other")
GENDER_OPTIONS = ("Female", "Male", "Unknown/Invalid")
MEDICATION_STATUS_OPTIONS = ("No", "Steady", "Up", "Down")
MAX_GLU_SERUM_OPTIONS = ("Norm", ">200", ">300")
A1C_RESULT_OPTIONS = ("Norm", ">7", ">8")
CHANGE_OPTIONS = ("No", "Ch")
DIABETES_MED_OPTIONS = ("No", "Yes")

AgeBucket = Literal[
    "[0-10)",
    "[10-20)",
    "[20-30)",
    "[30-40)",
    "[40-50)",
    "[50-60)",
    "[60-70)",
    "[70-80)",
    "[80-90)",
    "[90-100)",
]
WeightBucket = Literal[
    "[0-25)",
    "[25-50)",
    "[50-75)",
    "[75-100)",
    "[100-125)",
    "[125-150)",
    "[150-175)",
    "[175-200)",
    ">200",
]
RaceValue = Literal["AfricanAmerican", "Asian", "Caucasian", "Hispanic", "Other"]
GenderValue = Literal["Female", "Male", "Unknown/Invalid"]
MedicationStatus = Literal["No", "Steady", "Up", "Down"]
MaxGluSerumValue = Literal["Norm", ">200", ">300"]
A1CResultValue = Literal["Norm", ">7", ">8"]
ChangeValue = Literal["No", "Ch"]
DiabetesMedValue = Literal["No", "Yes"]

MEDICATION_FIELDS = (
    "metformin",
    "repaglinide",
    "nateglinide",
    "chlorpropamide",
    "glimepiride",
    "acetohexamide",
    "glipizide",
    "glyburide",
    "tolbutamide",
    "pioglitazone",
    "rosiglitazone",
    "acarbose",
    "miglitol",
    "troglitazone",
    "tolazamide",
    "examide",
    "citoglipton",
    "insulin",
    "glyburide-metformin",
    "glipizide-metformin",
    "glimepiride-pioglitazone",
    "metformin-rosiglitazone",
    "metformin-pioglitazone",
)


class PatientFeatures(BaseModel):
    model_config = ConfigDict(extra="forbid")

    encounter_id: int = Field(ge=0)
    patient_number: int = Field(ge=0)
    race: RaceValue | None = None
    gender: GenderValue
    age: AgeBucket
    weight: WeightBucket | None = None
    admission_type_id: int = Field(ge=0)
    discharge_disposition_id: int = Field(ge=0)
    admission_source_id: int = Field(ge=0)
    time_in_hospital: int = Field(ge=0)
    payer_code: str | None = None
    medical_specialty: str | None = None
    num_lab_procedures: int = Field(ge=0)
    num_procedures: int = Field(ge=0)
    num_medications: int = Field(ge=0)
    number_outpatient: int = Field(ge=0)
    number_emergency: int = Field(ge=0)
    number_inpatient: int = Field(ge=0)
    diag_1: str | None = None
    diag_2: str | None = None
    diag_3: str | None = None
    number_diagnoses: int = Field(ge=0)
    max_glu_serum: MaxGluSerumValue | None = None
    A1Cresult: A1CResultValue | None = None
    metformin: MedicationStatus
    repaglinide: MedicationStatus
    nateglinide: MedicationStatus
    chlorpropamide: MedicationStatus
    glimepiride: MedicationStatus
    acetohexamide: MedicationStatus
    glipizide: MedicationStatus
    glyburide: MedicationStatus
    tolbutamide: MedicationStatus
    pioglitazone: MedicationStatus
    rosiglitazone: MedicationStatus
    acarbose: MedicationStatus
    miglitol: MedicationStatus
    troglitazone: MedicationStatus
    tolazamide: MedicationStatus
    examide: MedicationStatus
    citoglipton: MedicationStatus
    insulin: MedicationStatus
    glyburide_metformin: MedicationStatus = Field(alias="glyburide-metformin")
    glipizide_metformin: MedicationStatus = Field(alias="glipizide-metformin")
    glimepiride_pioglitazone: MedicationStatus = Field(alias="glimepiride-pioglitazone")
    metformin_rosiglitazone: MedicationStatus = Field(alias="metformin-rosiglitazone")
    metformin_pioglitazone: MedicationStatus = Field(alias="metformin-pioglitazone")
    change: ChangeValue
    diabetesMed: DiabetesMedValue


class ShapContribution(BaseModel):
    feature: str
    impact: float


class PredictionResponse(BaseModel):
    probability: float
    prediction: int
    threshold_used: float
    shap_values: list[ShapContribution] | None = None
