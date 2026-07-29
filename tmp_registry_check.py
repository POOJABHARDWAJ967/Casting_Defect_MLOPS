import os
import mlflow
from mlflow import MlflowClient
import config

os.environ['MLFLOW_ALLOW_FILE_STORE'] = 'true'
mlflow.set_tracking_uri('file:./mlruns')
client = MlflowClient()
mv = client.get_model_version_by_alias(config.REGISTERED_MODEL, config.PRODUCTION_ALIAS)
print(f'Registered Model: {config.REGISTERED_MODEL}')
print(f'Production Alias: @{config.PRODUCTION_ALIAS}')
print(f'Version: {mv.version}')
print(f'Run ID: {mv.run_id}')
print(f'Status: {mv.status}')
print('\n=== Version History ===')
versions = client.search_model_versions(f"name='{config.REGISTERED_MODEL}'")
for v in versions:
    aliases = getattr(v, 'aliases', [])
    print(f"  v{v.version} — run={v.run_id[:8]}... aliases={aliases}")
