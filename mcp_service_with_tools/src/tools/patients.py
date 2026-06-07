import datetime

from src.db.connection import get_pool
from src.db import queries


def _age(dob: datetime.date | None) -> int | None:
    if dob is None:
        return None
    today = datetime.date.today()
    return today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))


def _patient_dict(row) -> dict:
    return {
        "id": str(row["id"]),
        "name": row["full_name"],
        "age": _age(row["date_of_birth"]),
        "phone": row["phone"],
        "address": {
            "line1": row["address_line1"],
            "line2": row["address_line2"],
            "city": row["city"],
            "state": row["state"],
            "postal_code": row["postal_code"],
            "country": row["country"],
        },
    }


async def get_all_patients() -> list[dict]:
    pool = get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(queries.GET_ALL_PATIENTS)
    return [_patient_dict(row) for row in rows]


async def get_patients_by_doctor(doctor_id: str) -> list[dict]:
    pool = get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(queries.GET_PATIENTS_BY_DOCTOR, doctor_id)
    return [_patient_dict(row) for row in rows]


async def get_medical_history(patient_id: str) -> list[dict]:
    pool = get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(queries.GET_MEDICAL_HISTORY, patient_id)
    return [
        {
            "id": str(row["id"]),
            "symptoms": row["symptoms"],
            "diagnosis": row["diagnosis"],
            "lab_results": row["lab_results"],
            "medication": row["medication_details"],
            "date": row["created_at"].isoformat(),
        }
        for row in rows
    ]
