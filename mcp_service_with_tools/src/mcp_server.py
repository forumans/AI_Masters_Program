from fastmcp import FastMCP
from src.tools import doctors, patients

mcp = FastMCP("Healthcare MCP Service")


@mcp.tool()
async def get_all_doctors() -> list[dict]:
    """Retrieve all doctors with their name, specialization, and contact info."""
    return await doctors.get_all_doctors()


@mcp.tool()
async def get_all_patients() -> list[dict]:
    """Retrieve all patients with their name, age, and contact info."""
    return await patients.get_all_patients()


@mcp.tool()
async def get_patients_by_doctor(doctor_id: str) -> list[dict]:
    """Given a doctor_id (UUID), retrieve all patients that doctor has treated with their name, age, and contact info."""
    return await patients.get_patients_by_doctor(doctor_id)


@mcp.tool()
async def get_doctors_by_patient(patient_id: str) -> list[dict]:
    """Given a patient_id (UUID), retrieve all doctors who have treated that patient with their name, specialization, and contact info."""
    return await doctors.get_doctors_by_patient(patient_id)


@mcp.tool()
async def get_medical_history(patient_id: str) -> list[dict]:
    """Given a patient_id (UUID), retrieve the full medical history including symptoms, diagnosis, lab results, and medication."""
    return await patients.get_medical_history(patient_id)


@mcp.tool()
async def get_doctor_availability(doctor_id: str) -> list[dict]:
    """Given a doctor_id (UUID), retrieve the availability slots for the next 30 days with their status (AVAILABLE, BOOKED, BLOCKED)."""
    return await doctors.get_doctor_availability(doctor_id)
