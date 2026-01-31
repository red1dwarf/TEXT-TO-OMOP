CREATE TABLE concept (
  concept_id integer NOT NULL,
  concept_name varchar(255) NOT NULL,
  domain_id varchar(20) NOT NULL,
  vocabulary_id varchar(20) NOT NULL,
  concept_class_id varchar(20) NOT NULL,
  standard_concept varchar(1) NULL,
  concept_code varchar(50) NOT NULL,
  valid_start_date date NOT NULL,
  valid_end_date date NOT NULL,
  invalid_reason varchar(1) NULL
);

CREATE TABLE condition_occurrence (
  condition_occurrence_id integer NOT NULL,
  person_id integer NOT NULL,
  condition_concept_id integer NOT NULL,
  condition_start_date date NOT NULL,
  condition_start_datetime TIMESTAMP NULL,
  condition_end_date date NULL,
  condition_end_datetime TIMESTAMP NULL,
  condition_type_concept_id integer NOT NULL,
  condition_status_concept_id integer NULL,
  stop_reason varchar(20) NULL,
  provider_id integer NULL,
  visit_occurrence_id integer NULL,
  visit_detail_id integer NULL,
  condition_source_value varchar(50) NULL,
  condition_source_concept_id integer NULL,
  condition_status_source_value varchar(50) NULL
);

CREATE TABLE visit_occurrence (
  visit_occurrence_id integer NOT NULL,
  person_id integer NOT NULL,
  visit_concept_id integer NOT NULL,
  visit_start_date date NOT NULL,
  visit_start_datetime TIMESTAMP NULL,
  visit_end_date date NOT NULL,
  visit_end_datetime TIMESTAMP NULL,
  visit_type_concept_id Integer NOT NULL,
  provider_id integer NULL,
  care_site_id integer NULL,
  visit_source_value varchar(50) NULL,
  visit_source_concept_id integer NULL,
  admitted_from_concept_id integer NULL,
  admitted_from_source_value varchar(50) NULL,
  discharged_to_concept_id integer NULL,
  discharged_to_source_value varchar(50) NULL,
  preceding_visit_occurrence_id integer NULL
);

CREATE TABLE drug_exposure (
  drug_exposure_id integer NOT NULL,
  person_id integer NOT NULL,
  drug_concept_id integer NOT NULL,
  drug_exposure_start_date date NOT NULL,
  drug_exposure_start_datetime TIMESTAMP NULL,
  drug_exposure_end_date date NOT NULL,
  drug_exposure_end_datetime TIMESTAMP NULL,
  verbatim_end_date date NULL,
  drug_type_concept_id integer NOT NULL,
  stop_reason varchar(20) NULL,
  refills integer NULL,
  quantity NUMERIC NULL,
  days_supply integer NULL,
  sig TEXT NULL,
  route_concept_id integer NULL,
  lot_number varchar(50) NULL,
  provider_id integer NULL,
  visit_occurrence_id integer NULL,
  visit_detail_id integer NULL,
  drug_source_value varchar(50) NULL,
  drug_source_concept_id integer NULL,
  route_source_value varchar(50) NULL,
  dose_unit_source_value varchar(50) NULL
);

CREATE TABLE measurement (
  measurement_id integer NOT NULL,
  person_id integer NOT NULL,
  measurement_concept_id integer NOT NULL,
  measurement_date date NOT NULL,
  measurement_datetime TIMESTAMP NULL,
  measurement_time varchar(10) NULL,
  measurement_type_concept_id integer NOT NULL,
  operator_concept_id integer NULL,
  value_as_number NUMERIC NULL,
  value_as_concept_id integer NULL,
  unit_concept_id integer NULL,
  range_low NUMERIC NULL,
  range_high NUMERIC NULL,
  provider_id integer NULL,
  visit_occurrence_id integer NULL,
  visit_detail_id integer NULL,
  measurement_source_value varchar(50) NULL,
  measurement_source_concept_id integer NULL,
  unit_source_value varchar(50) NULL,
  unit_source_concept_id integer NULL,
  value_source_value varchar(50) NULL,
  measurement_event_id bigint NULL,
  meas_event_field_concept_id integer NULL
);

CREATE TABLE visit_detail (
  visit_detail_id integer NOT NULL,
  person_id integer NOT NULL,
  visit_detail_concept_id integer NOT NULL,
  visit_detail_start_date date NOT NULL,
  visit_detail_start_datetime TIMESTAMP NULL,
  visit_detail_end_date date NOT NULL,
  visit_detail_end_datetime TIMESTAMP NULL,
  visit_detail_type_concept_id integer NOT NULL,
  provider_id integer NULL,
  care_site_id integer NULL,
  visit_detail_source_value varchar(50) NULL,
  visit_detail_source_concept_id Integer NULL,
  admitted_from_concept_id Integer NULL,
  admitted_from_source_value varchar(50) NULL,
  discharged_to_source_value varchar(50) NULL,
  discharged_to_concept_id integer NULL,
  preceding_visit_detail_id integer NULL,
  parent_visit_detail_id integer NULL,
  visit_occurrence_id integer NOT NULL
);

CREATE TABLE person (
  person_id integer NOT NULL,
  gender_concept_id integer NOT NULL,
  year_of_birth integer NOT NULL,
  month_of_birth integer NULL,
  day_of_birth integer NULL,
  birth_datetime TIMESTAMP NULL,
  race_concept_id integer NOT NULL,
  ethnicity_concept_id integer NOT NULL,
  location_id integer NULL,
  provider_id integer NULL,
  care_site_id integer NULL,
  person_source_value varchar(50) NULL,
  gender_source_value varchar(50) NULL,
  gender_source_concept_id integer NULL,
  race_source_value varchar(50) NULL,
  race_source_concept_id integer NULL,
  ethnicity_source_value varchar(50) NULL,
  ethnicity_source_concept_id integer NULL
);

CREATE TABLE device_exposure (
  device_exposure_id integer NOT NULL,
  person_id integer NOT NULL,
  device_concept_id integer NOT NULL,
  device_exposure_start_date date NOT NULL,
  device_exposure_start_datetime TIMESTAMP NULL,
  device_exposure_end_date date NULL,
  device_exposure_end_datetime TIMESTAMP NULL,
  device_type_concept_id integer NOT NULL,
  production_id varchar(255) NULL,
  quantity integer NULL,
  provider_id integer NULL,
  visit_occurrence_id integer NULL,
  visit_detail_id integer NULL,
  device_source_value varchar(50) NULL,
  device_source_concept_id integer NULL,
  unit_concept_id integer NULL,
  unit_source_value varchar(50) NULL,
  unit_source_concept_id integer NULL
);

CREATE TABLE procedure_occurrence (
  procedure_occurrence_id integer NOT NULL,
  person_id integer NOT NULL,
  procedure_concept_id integer NOT NULL,
  procedure_date date NOT NULL,
  procedure_datetime TIMESTAMP NULL,
  procedure_end_date date NULL,
  procedure_end_datetime TIMESTAMP NULL,
  procedure_type_concept_id integer NOT NULL,
  modifier_concept_id integer NULL,
  quantity integer NULL,
  provider_id integer NULL,
  visit_occurrence_id integer NULL,
  visit_detail_id integer NULL,
  procedure_source_value varchar(50) NULL,
  procedure_source_concept_id integer NULL,
  modifier_source_value varchar(50) NULL
);

CREATE TABLE specimen (
  specimen_id integer NOT NULL,
  person_id integer NOT NULL,
  specimen_concept_id integer NOT NULL,
  specimen_type_concept_id integer NOT NULL,
  specimen_date date NOT NULL,
  specimen_datetime TIMESTAMP NULL,
  quantity NUMERIC NULL,
  unit_concept_id integer NULL,
  anatomic_site_concept_id integer NULL,
  disease_status_concept_id integer NULL,
  specimen_source_id varchar(50) NULL,
  specimen_source_value varchar(50) NULL,
  unit_source_value varchar(50) NULL,
  anatomic_site_source_value varchar(50) NULL,
  disease_status_source_value varchar(50) NULL
);

CREATE TABLE note (
  note_id integer NOT NULL,
  person_id integer NOT NULL,
  note_date date NOT NULL,
  note_datetime TIMESTAMP NULL,
  note_type_concept_id integer NOT NULL,
  note_class_concept_id integer NOT NULL,
  note_title varchar(250) NULL,
  note_text TEXT NOT NULL,
  encoding_concept_id integer NOT NULL,
  language_concept_id integer NOT NULL,
  provider_id integer NULL,
  visit_occurrence_id integer NULL,
  visit_detail_id integer NULL,
  note_source_value varchar(50) NULL,
  note_event_id bigint NULL,
  note_event_field_concept_id integer NULL
);

CREATE TABLE note_nlp (
  note_nlp_id integer NOT NULL,
  note_id integer NOT NULL,
  section_concept_id integer NULL,
  snippet varchar(250) NULL,
  lexical_variant varchar(250) NOT NULL,
  note_nlp_concept_id integer NULL,
  note_nlp_source_concept_id integer NULL,
  nlp_system varchar(250) NULL,
  nlp_date date NOT NULL,
  nlp_datetime TIMESTAMP NULL,
  term_exists varchar(1) NULL,
  term_temporal varchar(50) NULL,
  term_modifiers varchar(2000) NULL
);

CREATE TABLE provider (
  provider_id integer NOT NULL,
  provider_name varchar(255) NULL,
  npi varchar(20) NULL,
  dea varchar(20) NULL,
  specialty_concept_id integer NULL,
  care_site_id integer NULL,
  year_of_birth integer NULL,
  gender_concept_id integer NULL,
  provider_source_value varchar(50) NULL,
  specialty_source_value varchar(50) NULL,
  specialty_source_concept_id integer NULL,
  gender_source_value varchar(50) NULL,
  gender_source_concept_id integer NULL
);

CREATE TABLE death (
  person_id integer NOT NULL,
  death_date date NOT NULL,
  death_datetime TIMESTAMP NULL,
  death_type_concept_id integer NULL,
  cause_concept_id integer NULL,
  cause_source_value varchar(50) NULL,
  cause_source_concept_id integer NULL
);

CREATE TABLE fact_relationship (
  domain_concept_id_1 integer NOT NULL,
  fact_id_1 integer NOT NULL,
  domain_concept_id_2 integer NOT NULL,
  fact_id_2 integer NOT NULL,
  relationship_concept_id integer NOT NULL
);

CREATE TABLE condition_era (
  condition_era_id integer NOT NULL,
  person_id integer NOT NULL,
  condition_concept_id integer NOT NULL,
  condition_era_start_date TIMESTAMP NOT NULL,
  condition_era_end_date TIMESTAMP NOT NULL,
  condition_occurrence_count integer NULL
);

CREATE TABLE episode (
  episode_id bigint NOT NULL,
  person_id bigint NOT NULL,
  episode_concept_id integer NOT NULL,
  episode_start_date date NOT NULL,
  episode_start_datetime TIMESTAMP NULL,
  episode_end_date date NULL,
  episode_end_datetime TIMESTAMP NULL,
  episode_parent_id bigint NULL,
  episode_number integer NULL,
  episode_object_concept_id integer NOT NULL,
  episode_type_concept_id integer NOT NULL,
  episode_source_value varchar(50) NULL,
  episode_source_concept_id integer NULL
);

CREATE TABLE episode_event (
  episode_id bigint NOT NULL,
  event_id bigint NOT NULL,
  episode_event_field_concept_id integer NOT NULL
);