from collections import namedtuple

DataIngestionArtifact = namedtuple("DataIngestionArtifact", 
["train_file_path","test_filepath","message","is_ingested"])

DataValidationArtifact = namedtuple("DataValidationArtifact",
["schema_file_path","report_file_path","report_page_file_path","is_validated","message"])