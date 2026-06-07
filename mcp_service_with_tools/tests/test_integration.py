"""
Integration tests — run each tool against the REAL database (DATABASE_URL).

Unlike test_tools.py (which mocks the DB), these tests connect for real and
print exactly what comes back, so you can see actual data in the terminal
(`pytest -s`) or in the HTML report's "Captured stdout" section per test.

Run only these with:  pytest -m integration -s
Skip these with:      pytest -m "not integration"
"""

import pytest
import pytest_asyncio

from src.db.connection import close_pool, init_pool
from src.tools import doctors, patients

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]


@pytest_asyncio.fixture
async def live_db():
    """Function-scoped: each test gets its own pool bound to its own event loop.

    asyncpg pools hold connections tied to the loop that created them, and
    pytest-asyncio gives each test function a fresh loop — a module/session
    scoped pool would be reused across loops and raise
    'Future attached to a different loop'.
    """
    await init_pool()
    yield
    await close_pool()


def _show(label: str, rows: list[dict]) -> None:
    print(f"\n--- {label} -> {len(rows)} row(s) ---")
    for row in rows:
        print(row)


async def test_get_all_doctors(live_db):
    result = await doctors.get_all_doctors()
    _show("get_all_doctors()", result)
    assert isinstance(result, list)


async def test_get_all_patients(live_db):
    result = await patients.get_all_patients()
    _show("get_all_patients()", result)
    assert isinstance(result, list)


async def test_get_patients_by_doctor(live_db):
    sample_doctor = (await doctors.get_all_doctors())[0]
    result = await patients.get_patients_by_doctor(sample_doctor["id"])
    _show(f"get_patients_by_doctor(doctor_id={sample_doctor['id']!r}  # {sample_doctor['name']})", result)
    assert isinstance(result, list)


async def test_get_doctors_by_patient(live_db):
    sample_patient = (await patients.get_all_patients())[0]
    result = await doctors.get_doctors_by_patient(sample_patient["id"])
    _show(f"get_doctors_by_patient(patient_id={sample_patient['id']!r}  # {sample_patient['name']})", result)
    assert isinstance(result, list)


async def test_get_medical_history(live_db):
    sample_patient = (await patients.get_all_patients())[0]
    result = await patients.get_medical_history(sample_patient["id"])
    _show(f"get_medical_history(patient_id={sample_patient['id']!r}  # {sample_patient['name']})", result)
    assert isinstance(result, list)


async def test_get_doctor_availability(live_db):
    sample_doctor = (await doctors.get_all_doctors())[0]
    result = await doctors.get_doctor_availability(sample_doctor["id"])
    _show(f"get_doctor_availability(doctor_id={sample_doctor['id']!r}  # {sample_doctor['name']})", result[:5])
    print(f"    ... ({len(result)} total slots in the next 30 days)")
    assert isinstance(result, list)
