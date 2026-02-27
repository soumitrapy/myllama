import os
import pandas
import pyspark
import gcsfs
import json
from pyspark.sql import SparkSession

# For gcsfs credential handling
from google.auth import default
from google.auth.transport.requests import Request
from google.oauth2 import service_account
import gcsfs


def get_cloud_path(filename):
  """Return the full cloud path for a given filename."""
  return f'gs://intlctlg-cdq/unified_qc/dev/s0d0jkl/{filename}'

def SaveInCloud(df, filename):
  """
  Save a pandas or Spark DataFrame to a cloud location in parquet format.

  Args:
    df: pandas.DataFrame or pyspark.sql.DataFrame
    filename: str, name of the file to save
  """
  path = get_cloud_path(filename)
  dir_path = os.path.dirname(path)
  os.makedirs(dir_path, exist_ok=True)

  if isinstance(df, list): # when it is a chat/alpaca format data
    assert path.endswith('.jsonl'), "For list data, please save as .jsonl file"
    # For vscode ssh remote: (in terminal) gcloud auth application-default login
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = os.path.expanduser('~/.config/gcloud/application_default_credentials.json')

    credentials, project = default()
    fs = gcsfs.GCSFileSystem(project=project, token=credentials)
    with fs.open(path, 'w') as f:
      for item in df:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f'data saved to {path}')
  
  elif isinstance(df, dict):
    assert path.endswith('.json'), "For dict data, please save as .json file"
    # For vscode ssh remote: (in terminal) gcloud auth application-default login
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = os.path.expanduser('~/.config/gcloud/application_default_credentials.json')

    credentials, project = default()
    fs = gcsfs.GCSFileSystem(project=project, token=credentials)
    with fs.open(path, 'w') as f:
      json.dump(df, f, ensure_ascii=False, indent=2)
    print(f'data saved to {path}')

  elif isinstance(df, pandas.DataFrame):
    # Ensure all object columns are string type for parquet compatibility
    for col in df.select_dtypes(include='object').columns:
      df[col] = df[col].astype(str)
    df.to_parquet(path, engine='pyarrow')
    print(f'DataFrame saved to {path}')

  elif isinstance(df, pyspark.sql.dataframe.DataFrame):
    df.write.mode('overwrite').parquet(path)
    print(f'Spark DataFrame saved to {path}')

  else:
    raise TypeError("df must be a pandas.DataFrame or pyspark.sql.DataFrame")

def LoadFromCloud(filename, spark=False):
  """
  Load a pandas or Spark DataFrame from a cloud location in parquet format.

  Args:
    filename: str, name of the file to load

  Returns:
    pandas.DataFrame or pyspark.sql.DataFrame
  """
  path = get_cloud_path(filename)
  if filename.endswith('.jsonl'):
    # For vscode ssh remote: (in terminal) gcloud auth application-default login
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = os.path.expanduser('~/.config/gcloud/application_default_credentials.json')

    credentials, project = default()
    fs = gcsfs.GCSFileSystem(project=project, token=credentials)
    data = []
    with fs.open(path, 'r') as f:
      for line in f:
        data.append(json.loads(line))
    print(f'List data loaded from {path}')
    return data
  elif filename.endswith('.json'):
    # For vscode ssh remote: (in terminal) gcloud auth application-default login
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = os.path.expanduser('~/.config/gcloud/application_default_credentials.json')

    credentials, project = default()
    fs = gcsfs.GCSFileSystem(project=project, token=credentials)
    with fs.open(path, 'r') as f:
      data = json.load(f)
    print(f'Dict data loaded from {path}')
    return data
  else:
    try:
      if spark:
        raise Exception("Forcing fallback to Spark DataFrame loading")
      # Try loading as pandas DataFrame
      df = pandas.read_parquet(path, engine='pyarrow')
      print(f'Pandas DataFrame loaded from {path}')
      return df
    except Exception as e:
      # Fallback to loading as Spark DataFrame
      spark = SparkSession.builder.getOrCreate()
      df = spark.read.parquet(path)
      print(f'Spark DataFrame loaded from {path}')
      return df


def upload_directory_with_gcsfs(local_dir, gcs_path, verbose=False):
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = os.path.expanduser('~/.config/gcloud/application_default_credentials.json')
    credentials, project = default()
    fs = gcsfs.GCSFileSystem(project=project, token=credentials)
    for root, _, files in os.walk(local_dir):
        for file in files:
            local_file = os.path.join(root, file)
            # Construct the GCS file path
            relative_path = os.path.relpath(local_file, local_dir)
            gcs_file_path = f"{gcs_path}/{relative_path}"
            with open(local_file, "rb") as fsrc:
                with fs.open(gcs_file_path, "wb") as fdst:
                    fdst.write(fsrc.read())
            if verbose:
                print(f"Uploaded {local_file} to {gcs_file_path}")

# # Example usage:
# upload_directory_with_gcsfs(
#     "../../checkpoints/llama-3.1-8B-Instruct-finetuned-ae/checkpoint-20000",
#     "intlctlg-cdq/unified_qc/dev/s0d0jkl/checkpoints/llama-3.1-8B-Instruct-finetuned-ae/checkpoint-20000"
# )