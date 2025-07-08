import os
import requests
import pandas as pd
from google import genai
from google.genai import types
import dotenv

dotenv.load_dotenv()

# === Configuration ===
# CKAN resource ID for Grants & Contributions
RESOURCE_ID = "1d15a62f-5656-49ad-8c88-f40ce689d831"
# CKAN API base URL
CKAN_API_BASE = "https://open.canada.ca/data/api/3/action/"
# Categories to use
CATEGORIES = [
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

# === Functions ===

def get_csv_url_from_resource(resource_id: str) -> str:
    """
    Fetches the direct CSV download URL for a given CKAN resource ID.
    """
    url = f"{CKAN_API_BASE}resource_show"
    resp = requests.get(url, params={"id": resource_id})
    resp.raise_for_status()
    result = resp.json().get("result", {})
    csv_url = result.get("url")
    if not csv_url or not csv_url.lower().endswith('.csv'):
        raise RuntimeError(f"Resource {resource_id} does not point to a CSV file.")
    return csv_url


def categorize_grant(title: str, description: str) -> str:
    """
    Uses an LLM to assign one category from the predefined list.
    """
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
        time.sleep(60//2000)

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

if __name__ == "__main__":
    '''
    COMPLETE

    # 1. Resolve the CSV URL from the specified resource
    print(f"Fetching CSV URL for resource {RESOURCE_ID}...")
    csv_url = get_csv_url_from_resource(RESOURCE_ID)
    print(f"Loading grants data from: {csv_url}")

    # 2. Load the grants data
    df = pd.read_csv(csv_url)

    # 2.5 save the raw data for reference
    raw_data_file = 'raw_grants_data.csv'
    df.to_csv(raw_data_file, index=False)'''

    # Try to perform basic cleaning
    try:
        df = pd.read_csv("raw_grants_data.csv")

        print("Raw data loaded successfully.")
        print(f"Data contains {len(df)} rows and {len(df.columns)} columns.")
        print("Renaming columns and selecting relevant ones...")

        df = df.rename(columns={
            'prog_name_en': 'title',
            'agreement_value': 'value',
            'agreement_title_en': 'agreement_title',
            'description_en': 'description',
            'recipient_legal_name': 'recipient'
        })[[ 'title', 'agreement_title', 'description', 'recipient', 'value' ]]

        print("Columns renamed and selected.")
        print("Sampling and categorizing grants...")

        # 3. Sample and categorize (adjust sample size as needed)
        sample = df.sample(n=100000, random_state=42).reset_index(drop=True)
        sample['category'] = sample.apply(
            lambda row: categorize_grant(row['title'], row['description']), axis=1
        )

        print("Sample categorized successfully.")
        print(f"Sample contains {len(sample)} rows and {len(sample.columns)} columns.")

        # 4. Save the categorized sample
        output_file = 'categorized_grants_sample.csv'
        sample.to_csv(output_file, index=False)
        print(f"Categorized sample saved to '{output_file}'.")
    except Exception as e:
        print(f"An error occurred while processing the data: {e}")
        print("Please check the raw data file for more details.")
