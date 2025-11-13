-- Optional: create and use a dedicated schema
CREATE DATABASE IF NOT EXISTS health_insurance CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci;
USE health_insurance;

-- INSURANCE
CREATE TABLE `INSURANCE` (
  `insurance_id` BIGINT UNSIGNED NOT NULL,
  `insurance_name` VARCHAR(255) NOT NULL,
  `contact_info` JSON NULL,
  PRIMARY KEY (`insurance_id`),
  UNIQUE KEY `ux_insurance_name` (`insurance_name`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- PLAN
CREATE TABLE `PLAN` (
  `plan_id` BIGINT UNSIGNED NOT NULL,
  `name` VARCHAR(255) NOT NULL,
  `insurance_id` BIGINT UNSIGNED NOT NULL,
  `base_rate` DECIMAL(12,2) NOT NULL,
  `deductable` DECIMAL(12,2) NOT NULL,
  PRIMARY KEY (`plan_id`),
  KEY `ix_plan_insurance_id` (`insurance_id`),
  CONSTRAINT `fk_plan_insurance`
    FOREIGN KEY (`insurance_id`) REFERENCES `INSURANCE` (`insurance_id`)
    ON DELETE RESTRICT ON UPDATE CASCADE,
  CONSTRAINT `ck_plan_base_rate_nonneg` CHECK (`base_rate` >= 0),
  CONSTRAINT `ck_plan_deductable_nonneg` CHECK (`deductable` >= 0),
  UNIQUE KEY `ux_plan_insurance_name` (`insurance_id`, `name`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- FACILITY
CREATE TABLE `FACILITY` (
  `Facility_ID` BIGINT UNSIGNED NOT NULL,
  `NPI` CHAR(10) NULL,
  `Name` VARCHAR(255) NOT NULL,
  `State` CHAR(2) NOT NULL,
  `Contact_info` JSON NULL,
  PRIMARY KEY (`Facility_ID`),
  KEY `ix_facility_state` (`State`),
  KEY `ix_facility_npi` (`NPI`),
  CONSTRAINT `ck_facility_state` CHECK (REGEXP_LIKE(`State`, '^[A-Z]{2}$'))
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- ENROLLMENT
CREATE TABLE `ENROLLMENT` (
  `Enrollment_ID` BIGINT UNSIGNED NOT NULL,
  `Coverage_tier` VARCHAR(50) NOT NULL,
  `start_date` DATE NOT NULL,
  `end_date` DATE NULL,
  `state` CHAR(2) NOT NULL,
  `plan_id` BIGINT UNSIGNED NOT NULL,
  `premium` DECIMAL(12,2) NOT NULL,
  PRIMARY KEY (`Enrollment_ID`),
  KEY `ix_enrollment_plan_id` (`plan_id`),
  KEY `ix_enrollment_state` (`state`),
  CONSTRAINT `fk_enrollment_plan`
    FOREIGN KEY (`plan_id`) REFERENCES `PLAN` (`plan_id`)
    ON DELETE RESTRICT ON UPDATE CASCADE,
  CONSTRAINT `ck_enrollment_premium_nonneg` CHECK (`premium` >= 0),
  CONSTRAINT `ck_enrollment_dates` CHECK (`end_date` IS NULL OR `end_date` >= `start_date`),
  CONSTRAINT `ck_enrollment_state` CHECK (REGEXP_LIKE(`state`, '^[A-Z]{2}$'))
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- CONDITION
CREATE TABLE `CONDITION` (
  `Condition_ID` BIGINT UNSIGNED NOT NULL,
  `Condition_name` VARCHAR(255) NOT NULL,
  PRIMARY KEY (`Condition_ID`),
  UNIQUE KEY `ux_condition_name` (`Condition_name`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- MEMBERS
CREATE TABLE `MEMBERS` (
  `member_id` BIGINT UNSIGNED NOT NULL,
  `DOB` DATE NOT NULL,
  `Sex` ENUM('F','M','O') NOT NULL,
  `Primary_Care_Facility_ID` BIGINT UNSIGNED NULL,
  `State` CHAR(2) NOT NULL,
  `Weight` DECIMAL(6,2) NULL,
  `Height` DECIMAL(5,2) NULL,
  `heart_rate` SMALLINT UNSIGNED NULL,
  `blood_pressure` VARCHAR(15) NULL,
  `blood_oxygen` TINYINT UNSIGNED NULL,
  `smoker` TINYINT(1) NULL,
  `drinker` TINYINT(1) NULL,
  `housing insecurity` TINYINT(1) NULL,
  `employment status` TINYINT(1) NULL,
  `hours_sleep_per_day` DECIMAL(4,1) NULL,
  `minutes_exercise_per_week` SMALLINT UNSIGNED NULL,
  `Insurance ID` BIGINT UNSIGNED NULL,
  `Enrollment ID` BIGINT UNSIGNED NULL,
  PRIMARY KEY (`member_id`),
  KEY `ix_members_state` (`State`),
  KEY `ix_members_facility` (`Primary_Care_Facility_ID`),
  KEY `ix_members_insurance` (`Insurance ID`),
  KEY `ix_members_enrollment` (`Enrollment ID`),
  CONSTRAINT `fk_members_facility`
    FOREIGN KEY (`Primary_Care_Facility_ID`) REFERENCES `FACILITY` (`Facility_ID`)
    ON DELETE SET NULL ON UPDATE CASCADE,
  CONSTRAINT `fk_members_insurance`
    FOREIGN KEY (`Insurance ID`) REFERENCES `INSURANCE` (`insurance_id`)
    ON DELETE SET NULL ON UPDATE CASCADE,
  CONSTRAINT `fk_members_enrollment`
    FOREIGN KEY (`Enrollment ID`) REFERENCES `ENROLLMENT` (`Enrollment_ID`)
    ON DELETE SET NULL ON UPDATE CASCADE,
  CONSTRAINT `ck_members_state` CHECK (REGEXP_LIKE(`State`, '^[A-Z]{2}$')),
  CONSTRAINT `ck_members_weight_pos` CHECK (`Weight` IS NULL OR `Weight` > 0),
  CONSTRAINT `ck_members_height_pos` CHECK (`Height` IS NULL OR `Height` > 0),
  CONSTRAINT `ck_members_hr_valid` CHECK (`heart_rate` IS NULL OR (`heart_rate` BETWEEN 20 AND 300)),
  CONSTRAINT `ck_members_bp_pattern` CHECK (`blood_pressure` IS NULL OR REGEXP_LIKE(`blood_pressure`, '^[0-9]{2,3}/[0-9]{2,3}$')),
  CONSTRAINT `ck_members_spo2_valid` CHECK (`blood_oxygen` IS NULL OR (`blood_oxygen` BETWEEN 0 AND 100)),
  CONSTRAINT `ck_members_smoker_bool` CHECK (`smoker` IN (0,1) OR `smoker` IS NULL),
  CONSTRAINT `ck_members_drinker_bool` CHECK (`drinker` IN (0,1) OR `drinker` IS NULL),
  CONSTRAINT `ck_members_housing_bool` CHECK (`housing insecurity` IN (0,1) OR `housing insecurity` IS NULL),
  CONSTRAINT `ck_members_employ_bool` CHECK (`employment status` IN (0,1) OR `employment status` IS NULL),
  CONSTRAINT `ck_members_sleep_range` CHECK (`hours_sleep_per_day` IS NULL OR (`hours_sleep_per_day` BETWEEN 0 AND 24)),
  CONSTRAINT `ck_members_exercise_range` CHECK (`minutes_exercise_per_week` IS NULL OR (`minutes_exercise_per_week` <= 10080))
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- MEMBER_CONDITION
CREATE TABLE `MEMBER_CONDITION` (
  `Member_ID` BIGINT UNSIGNED NOT NULL,
  `Condition_ID` BIGINT UNSIGNED NOT NULL,
  `Diagnostic_date` DATE NOT NULL,
  PRIMARY KEY (`Member_ID`, `Condition_ID`, `Diagnostic_date`),
  KEY `ix_member_condition_condition` (`Condition_ID`),
  KEY `ix_member_condition_diagnostic_date` (`Diagnostic_date`),
  CONSTRAINT `fk_member_condition_member`
    FOREIGN KEY (`Member_ID`) REFERENCES `MEMBERS` (`member_id`)
    ON DELETE CASCADE ON UPDATE CASCADE,
  CONSTRAINT `fk_member_condition_condition`
    FOREIGN KEY (`Condition_ID`) REFERENCES `CONDITION` (`Condition_ID`)
    ON DELETE RESTRICT ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- CLAIMS
CREATE TABLE `CLAIMS` (
  `claim_id` BIGINT UNSIGNED NOT NULL,
  `member_id` BIGINT UNSIGNED NOT NULL,
  `amount` DECIMAL(12,2) NOT NULL,
  `date` DATE NOT NULL,
  `insurance_id` BIGINT UNSIGNED NOT NULL,
  PRIMARY KEY (`claim_id`),
  KEY `ix_claims_member_date` (`member_id`, `date`),
  KEY `ix_claims_insurance_date` (`insurance_id`, `date`),
  CONSTRAINT `fk_claims_member`
    FOREIGN KEY (`member_id`) REFERENCES `MEMBERS` (`member_id`)
    ON DELETE RESTRICT ON UPDATE CASCADE,
  CONSTRAINT `fk_claims_insurance`
    FOREIGN KEY (`insurance_id`) REFERENCES `INSURANCE` (`insurance_id`)
    ON DELETE RESTRICT ON UPDATE CASCADE,
  CONSTRAINT `ck_claims_amount_nonneg` CHECK (`amount` >= 0)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;