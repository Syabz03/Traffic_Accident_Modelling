import os

settings = {
    'host': os.environ.get('ACCOUNT_HOST', 'https://tam-23.documents.azure.com:443/'),
    'master_key': os.environ.get('ACCOUNT_KEY', 'fY73GPzCA4RiCWVOPDO8MNCbA24u0Wq9LE3q36x4260cjLBrVJM5EIEvSoytZFtMM2NrRHEngGLmACDbL72j6w=='),
    'database_id': os.environ.get('COSMOS_DATABASE', 'tam'),
    'accidents_container_id': os.environ.get('COSMOS_CONTAINER', 'USAccidents'),
    'defaults_container_id': os.environ.get('COSMOS_CONTAINER', 'defaults'),
}