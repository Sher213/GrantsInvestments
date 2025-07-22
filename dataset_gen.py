import os
import io
import logging
import urllib3
import requests
import pandas as pd
import asyncio
from concurrent.futures import ProcessPoolExecutor
from google import genai
from google.genai import types
import dotenv

# ——— Logging setup —————————————————————————————————————————————
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(processName)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# Silence InsecureRequestWarning when verify=False
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ——— Configuration —————————————————————————————————————————————
dotenv.load_dotenv()
RESOURCE_ID   = "1d15a62f-5656-49ad-8c88-f40ce689d831"
CKAN_API_BASE = "https://open.canada.ca/data/api/3/action/"
CATEGORIES    = [
    "Housing & Shelter",
    "Education & Training",
    "Employment & Entrepreneurship",
    "Business & Innovation",
    "Health & Wellness",
    "Environment & Energy",
    "Community & Nonprofits",
    "Research & Academia",
    "Indigenous Programs",
    "Public Safety & Emergency Services",
    "Agriculture & Rural Development",
    "Arts, Culture & Heritage",
    "Civic & Democratic Engagement"
]

def get_csv_url_from_resource(resource_id: str) -> str:
    logger.info(f"Fetching CSV URL for resource {resource_id}")
    resp = requests.get(f"{CKAN_API_BASE}resource_show", params={"id": resource_id})
    resp.raise_for_status()
    url = resp.json()["result"]["url"]
    if not url.lower().endswith(".csv"):
        logger.error("Resource is not a CSV: %s", url)
        raise RuntimeError(f"Resource {resource_id} is not a CSV.")
    logger.info("Found CSV URL: %s", url)
    return url

def fetch_csv_via_requests(url: str) -> pd.DataFrame:
    logger.info("Downloading CSV via requests (verify=False)")
    resp = requests.get(url, verify=False)
    resp.raise_for_status()
    logger.info("Download complete, parsing into DataFrame")
    return pd.read_csv(io.StringIO(resp.text))

def categorize_grant(title: str, description: str) -> str:
    logger.debug("Categorizing grant: %.50s…", title)
    try:
        client = genai.Client(
            api_key=os.getenv("GEMINI_API_KEY"),
        )

        model = "gemini-2.0-flash"
        contents = [
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(text=f"""
    You are a helpful assistant that categorizes government grants for citizens.
    Grant Title: {title}
    Description: {description}

    Choose exactly one category from the list below:
    {chr(10).join(CATEGORIES)}
    Respond with the category name only.
    """),
                ],
            ),
        ]

        # add delay to avoid rate limiting
        import time
        time.sleep(60/2000)

        generate_content_config = types.GenerateContentConfig(
            response_mime_type="text/plain",
            system_instruction=[
                types.Part.from_text(text="Categorize Canadian government grants into citizen-friendly labels."),
            ],
        )

        resp = ""

        # Stream the response from the model
        for chunk in client.models.generate_content_stream(
            model=model,
            contents=contents,
            config=generate_content_config,
        ):
            resp += chunk.text

        print(f"Categorized grant '{title}' as: {resp.strip()}")
        return resp
    except Exception as e:
        print(f"Error categorizing grant '{title}': {e}")
        return "Uncategorized"

    except Exception:
        logger.exception("Error categorizing grant: %s", title)
        return "Uncategorized"

async def main():
    raw_csv_path = "raw_grants_data.csv"

    # 1. Fetch or load raw CSV
    if not os.path.exists(raw_csv_path):
        url = get_csv_url_from_resource(RESOURCE_ID)
        df = fetch_csv_via_requests(url)
        df.to_csv(raw_csv_path, index=False)
        logger.info("Raw CSV saved to %s", raw_csv_path)
    else:
        logger.info("Loading raw CSV from %s", raw_csv_path)
        df = pd.read_csv(raw_csv_path)

    # 2. Clean & select
    df = df.rename(columns={
        'prog_name_en':        'title',
        'agreement_title_en':  'agreement_title',
        'description_en':      'description',
        'recipient_legal_name':'recipient',
        'agreement_value':     'value'
    })[['title','agreement_title','description','recipient','value']]
    logger.info("DataFrame cleaned; %d rows × %d columns", *df.shape)

    # 3. Sample
    sample = df.sample(n=100_000, random_state=42).reset_index(drop=True)
    logger.info("Sampled %d rows", len(sample))

    # 4. Parallel categorize
    loop = asyncio.get_running_loop()
    with ProcessPoolExecutor() as pool:
        logger.info("Starting categorization with %d workers", pool._max_workers)
        tasks = [
            loop.run_in_executor(pool, categorize_grant, title, desc)
            for title, desc in sample[['title','description']].itertuples(index=False, name=None)
        ]
        categories = await asyncio.gather(*tasks)

    sample['category'] = categories

    # 5. Save
    output_file = "categorized_grants_sample.csv"
    sample.to_csv(output_file, index=False)
    logger.info("Categorized sample written to %s", output_file)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except:
        logger.exception("Fatal error in main()")
        raise
