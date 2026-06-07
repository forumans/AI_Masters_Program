import datetime
import uuid
import pytest
from unittest.mock import patch

DOCTOR_ID  = str(uuid.uuid4())
PATIENT_ID = str(uuid.uuid4())
RECORD_ID  = str(uuid.uuid4())


# ---------------------------------------------------------------------------
# Doctors
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_get_all_doctors(mock_pool_and_conn):
    pool, conn = mock_pool_and_conn
    conn.fetch.return_value = [
        {"id": uuid.UUID(DOCTOR_ID), "full_name": "Dr. Smith", "specialty": "Cardiology", "phone": "555-0001"},
    ]

    with patch("src.tools.doctors.get_pool", return_value=pool):
        from src.tools.doctors import get_all_doctors
        result = await get_all_doctors()

    assert len(result) == 1
    assert result[0]["name"] == "Dr. Smith"
    assert result[0]["specialization"] == "Cardiology"
    assert result[0]["id"] == DOCTOR_ID


@pytest.mark.asyncio
async def test_get_doctor_availability(mock_pool_and_conn):
    pool, conn = mock_pool_and_conn
    slot = datetime.datetime(2026, 6, 10, 9, 0)
    conn.fetch.return_value = [
        {"id": uuid.UUID(DOCTOR_ID), "slot_time": slot, "status": "AVAILABLE", "block_reason": None},
    ]

    with patch("src.tools.doctors.get_pool", return_value=pool):
        from src.tools.doctors import get_doctor_availability
        result = await get_doctor_availability(DOCTOR_ID)

    assert len(result) == 1
    assert result[0]["slot_time"] == slot.isoformat()
    assert result[0]["status"] == "AVAILABLE"
    assert result[0]["block_reason"] is None


@pytest.mark.asyncio
async def test_get_doctors_by_patient(mock_pool_and_conn):
    pool, conn = mock_pool_and_conn
    conn.fetch.return_value = [
        {"id": uuid.UUID(DOCTOR_ID), "full_name": "Dr. Chen", "specialty": "Neurology", "phone": "555-0002"},
    ]

    with patch("src.tools.doctors.get_pool", return_value=pool):
        from src.tools.doctors import get_doctors_by_patient
        result = await get_doctors_by_patient(PATIENT_ID)

    assert result[0]["specialization"] == "Neurology"


# ---------------------------------------------------------------------------
# Patients
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_get_all_patients(mock_pool_and_conn):
    pool, conn = mock_pool_and_conn
    dob = datetime.date(1981, 3, 25)
    conn.fetch.return_value = [
        {
            "id": uuid.UUID(PATIENT_ID), "full_name": "Alice Thompson",
            "date_of_birth": dob, "phone": "555-1001",
            "address_line1": "123 Main St", "address_line2": None,
            "city": "Ashburn", "state": "Virginia", "postal_code": "20147", "country": "US",
        },
    ]

    with patch("src.tools.patients.get_pool", return_value=pool):
        from src.tools.patients import get_all_patients
        result = await get_all_patients()

    assert len(result) == 1
    assert result[0]["name"] == "Alice Thompson"
    assert result[0]["age"] == datetime.date.today().year - 1981 - (
        (datetime.date.today().month, datetime.date.today().day) < (3, 25)
    )
    assert result[0]["address"]["city"] == "Ashburn"


@pytest.mark.asyncio
async def test_get_patients_by_doctor(mock_pool_and_conn):
    pool, conn = mock_pool_and_conn
    conn.fetch.return_value = [
        {
            "id": uuid.UUID(PATIENT_ID), "full_name": "Bob Martinez",
            "date_of_birth": datetime.date(1962, 7, 22), "phone": "555-1002",
            "address_line1": None, "address_line2": None,
            "city": "Herndon", "state": "Virginia", "postal_code": None, "country": "US",
        },
    ]

    with patch("src.tools.patients.get_pool", return_value=pool):
        from src.tools.patients import get_patients_by_doctor
        result = await get_patients_by_doctor(DOCTOR_ID)

    assert result[0]["name"] == "Bob Martinez"


@pytest.mark.asyncio
async def test_get_medical_history(mock_pool_and_conn):
    pool, conn = mock_pool_and_conn
    created = datetime.datetime(2025, 1, 15, 10, 0)
    conn.fetch.return_value = [
        {
            "id": uuid.UUID(RECORD_ID),
            "symptoms": "Chest tightness",
            "diagnosis": "Hypertension",
            "lab_results": "BP elevated",
            "medication_details": "Lisinopril 10mg daily",
            "created_at": created,
        }
    ]

    with patch("src.tools.patients.get_pool", return_value=pool):
        from src.tools.patients import get_medical_history
        result = await get_medical_history(PATIENT_ID)

    assert result[0]["diagnosis"] == "Hypertension"
    assert result[0]["medication"] == "Lisinopril 10mg daily"
    assert result[0]["date"] == created.isoformat()


@pytest.mark.asyncio
async def test_medical_history_no_prescription(mock_pool_and_conn):
    """A medical record with no linked prescription returns medication=None."""
    pool, conn = mock_pool_and_conn
    conn.fetch.return_value = [
        {
            "id": uuid.UUID(RECORD_ID),
            "symptoms": "Headache",
            "diagnosis": "Tension headache",
            "lab_results": None,
            "medication_details": None,
            "created_at": datetime.datetime(2025, 3, 10, 9, 0),
        }
    ]

    with patch("src.tools.patients.get_pool", return_value=pool):
        from src.tools.patients import get_medical_history
        result = await get_medical_history(PATIENT_ID)

    assert result[0]["medication"] is None


# ---------------------------------------------------------------------------
# Pool not initialized guard
# ---------------------------------------------------------------------------

def test_get_pool_raises_before_init():
    import src.db.connection as db_module
    original = db_module._pool
    db_module._pool = None

    with pytest.raises(RuntimeError, match="not initialized"):
        db_module.get_pool()

    db_module._pool = original
