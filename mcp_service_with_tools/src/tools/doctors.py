from src.db.connection import get_pool
from src.db import queries


async def get_all_doctors() -> list[dict]:
    pool = get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(queries.GET_ALL_DOCTORS)
    return [
        {
            "id": str(row["id"]),
            "name": row["full_name"],
            "specialization": row["specialty"],
            "phone": row["phone"],
        }
        for row in rows
    ]


async def get_doctor_availability(doctor_id: str) -> list[dict]:
    pool = get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(queries.GET_DOCTOR_AVAILABILITY, doctor_id)
    return [
        {
            "id": str(row["id"]),
            "slot_time": row["slot_time"].isoformat(),
            "status": str(row["status"]),
            "block_reason": row["block_reason"],
        }
        for row in rows
    ]


async def get_doctors_by_patient(patient_id: str) -> list[dict]:
    pool = get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(queries.GET_DOCTORS_BY_PATIENT, patient_id)
    return [
        {
            "id": str(row["id"]),
            "name": row["full_name"],
            "specialization": row["specialty"],
            "phone": row["phone"],
        }
        for row in rows
    ]
