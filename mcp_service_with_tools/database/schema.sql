-- Requires the uuid-ossp extension for uuid_generate_v4()
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Enums
CREATE TYPE user_role          AS ENUM ('ADMIN', 'DOCTOR', 'PATIENT');
CREATE TYPE appointment_status AS ENUM ('SCHEDULED', 'CONFIRMED', 'CANCELLED', 'COMPLETED');
CREATE TYPE slot_status        AS ENUM ('AVAILABLE', 'BOOKED', 'BLOCKED');
CREATE TYPE gender_type        AS ENUM ('MALE', 'FEMALE', 'OTHER', 'PREFER_NOT_TO_SAY');

CREATE TABLE tenants (
    id         UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name       VARCHAR(255) NOT NULL,
    domain     VARCHAR(255),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE users (
    id            UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id     UUID         NOT NULL REFERENCES tenants(id),
    email         VARCHAR(255) NOT NULL,
    password_hash TEXT         NOT NULL,
    role          user_role    NOT NULL,
    is_active     BOOLEAN      DEFAULT TRUE,
    created_at    TIMESTAMP    DEFAULT CURRENT_TIMESTAMP,
    deleted_at    TIMESTAMP,
    deleted_by    UUID
);

CREATE TABLE doctors (
    id             UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id      UUID         NOT NULL REFERENCES tenants(id),
    user_id        UUID         NOT NULL REFERENCES users(id),
    full_name      VARCHAR(255) NOT NULL,
    specialty      VARCHAR(255),
    license_number VARCHAR(255),
    phone          VARCHAR(50),
    date_of_birth  DATE,
    created_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    deleted_at     TIMESTAMP,
    deleted_by     UUID
);

CREATE TABLE patients (
    id                     UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id              UUID         NOT NULL REFERENCES tenants(id),
    user_id                UUID         NOT NULL REFERENCES users(id),
    full_name              VARCHAR(255) NOT NULL,
    phone                  VARCHAR(50),
    date_of_birth          DATE,
    gender                 VARCHAR(50),
    insurance_provider     VARCHAR(255),
    insurance_policy_number VARCHAR(255),
    address_line1          VARCHAR(255),
    address_line2          VARCHAR(255),
    city                   VARCHAR(100),
    state                  VARCHAR(100),
    postal_code            VARCHAR(20),
    country                VARCHAR(100),
    created_at             TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    deleted_at             TIMESTAMP,
    deleted_by             UUID
);

CREATE TABLE appointments (
    id               UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id        UUID               NOT NULL REFERENCES tenants(id),
    doctor_id        UUID               NOT NULL,
    patient_id       UUID               NOT NULL,
    slot_id          UUID,
    appointment_time TIMESTAMP          NOT NULL,
    status           appointment_status DEFAULT 'SCHEDULED',
    notes            TEXT,
    created_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    deleted_at       TIMESTAMP,
    deleted_by       UUID
);

CREATE TABLE doctor_availability (
    id           UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id    UUID        NOT NULL REFERENCES tenants(id),
    doctor_id    UUID        NOT NULL,
    slot_time    TIMESTAMP   NOT NULL,
    status       slot_status DEFAULT 'AVAILABLE',
    block_reason TEXT,
    created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE medical_records (
    id             UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id      UUID NOT NULL REFERENCES tenants(id),
    appointment_id UUID NOT NULL,
    symptoms       TEXT,
    diagnosis      TEXT,
    lab_results    TEXT,
    created_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    deleted_at     TIMESTAMP,
    deleted_by     UUID
);

CREATE TABLE prescriptions (
    id                UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id         UUID NOT NULL REFERENCES tenants(id),
    medical_record_id UUID NOT NULL,
    pharmacy_id       UUID,
    medication_details TEXT NOT NULL,
    created_at        TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    deleted_at        TIMESTAMP,
    deleted_by        UUID
);

CREATE TABLE pharmacies (
    id         UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id  UUID         NOT NULL REFERENCES tenants(id),
    name       VARCHAR(255) NOT NULL,
    address    TEXT,
    phone      TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    deleted_at TIMESTAMP,
    deleted_by UUID
);

CREATE TABLE admins (
    id         UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id  UUID NOT NULL REFERENCES tenants(id),
    user_id    UUID,
    full_name  VARCHAR(255),
    phone      VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    deleted_at TIMESTAMP,
    deleted_by UUID
);

CREATE TABLE audit_logs (
    id           UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id    UUID,
    table_name   TEXT      NOT NULL,
    record_id    UUID      NOT NULL,
    action_type  TEXT      NOT NULL,
    old_data     JSONB,
    new_data     JSONB,
    performed_by UUID,
    performed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for common query patterns
CREATE INDEX idx_appointments_doctor_id   ON appointments(doctor_id);
CREATE INDEX idx_appointments_patient_id  ON appointments(patient_id);
CREATE INDEX idx_availability_doctor_time ON doctor_availability(doctor_id, slot_time);
CREATE INDEX idx_medical_records_appt_id  ON medical_records(appointment_id);
CREATE INDEX idx_prescriptions_record_id  ON prescriptions(medical_record_id);
