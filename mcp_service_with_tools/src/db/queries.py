GET_ALL_DOCTORS = """
    SELECT id, full_name, specialty, phone
    FROM doctors
    WHERE deleted_at IS NULL
    ORDER BY full_name
"""

GET_ALL_PATIENTS = """
    SELECT id, full_name, date_of_birth, phone,
           address_line1, address_line2, city, state, postal_code, country
    FROM patients
    WHERE deleted_at IS NULL
    ORDER BY full_name
"""

GET_PATIENTS_BY_DOCTOR = """
    SELECT DISTINCT
        p.id, p.full_name, p.date_of_birth, p.phone,
        p.address_line1, p.address_line2, p.city, p.state, p.postal_code, p.country
    FROM patients p
    JOIN appointments a ON p.id = a.patient_id
    WHERE a.doctor_id = $1
      AND p.deleted_at IS NULL
      AND a.deleted_at IS NULL
    ORDER BY p.full_name
"""

GET_DOCTORS_BY_PATIENT = """
    SELECT DISTINCT d.id, d.full_name, d.specialty, d.phone
    FROM doctors d
    JOIN appointments a ON d.id = a.doctor_id
    WHERE a.patient_id = $1
      AND d.deleted_at IS NULL
      AND a.deleted_at IS NULL
    ORDER BY d.full_name
"""

GET_MEDICAL_HISTORY = """
    SELECT
        mr.id,
        mr.symptoms,
        mr.diagnosis,
        mr.lab_results,
        pr.medication_details,
        mr.created_at
    FROM medical_records mr
    JOIN appointments a ON mr.appointment_id = a.id
    LEFT JOIN prescriptions pr
        ON pr.medical_record_id = mr.id AND pr.deleted_at IS NULL
    WHERE a.patient_id = $1
      AND mr.deleted_at IS NULL
    ORDER BY mr.created_at DESC
"""

GET_DOCTOR_AVAILABILITY = """
    SELECT id, slot_time, status, block_reason
    FROM doctor_availability
    WHERE doctor_id = $1
      AND slot_time >= NOW()
      AND slot_time < NOW() + INTERVAL '30 days'
    ORDER BY slot_time
"""
