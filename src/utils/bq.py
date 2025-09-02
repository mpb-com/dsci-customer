from google.cloud import bigquery
import dotenv
import os
import pandas as pd


class BQ:
    def __init__(self, project_id=None):
        dotenv.load_dotenv()
        self.client = bigquery.Client(project=project_id or os.getenv("GCP_PROJECT_ID"))

    def to_dataframe(
        self,
        query: str,
        job_config: bigquery.QueryJobConfig = None,
        dtypes: dict = None,
    ) -> pd.DataFrame:
        """
        Execute a query and return the results as a pandas DataFrame.
        """
        df = self.client.query(query, job_config=job_config).result().to_dataframe()
        if dtypes:
            df = df.astype(dtypes)
        return df
