use reqwest::Client;
use scraper::{Html, Selector};
use tokio;
use csv::Writer;
use chrono::{Utc, Duration, NaiveDate, DateTime};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    //  – compute the cutoff
    let one_month_ago = Utc::now() - Duration::days(120);

    //  – CSV setup
    let mut wtr = Writer::from_path("../pulled_grants.csv")?;
    wtr.write_record(&[
        "Agreement",
        "Agreement Number",
        "Date Agreement",
        "Date Agreed",
        "Description",
        "Recipient",
        "Recipient Public Name",
        "Price",
        "Location",
    ])?;

    //  – selectors
    let item_sel    = Selector::parse("div.row.mrgn-bttm-xl.mrgn-lft-md").unwrap();
    let info_sel    = Selector::parse("div.col-sm-12.mrgn-tp-0").unwrap();
    let generic_sel = Selector::parse("div.col-sm-12").unwrap();
    let name_sel    = Selector::parse("div.col-sm-8").unwrap();
    let price_sel   = Selector::parse("div.col-sm-4.text-right h4.mrgn-tp-0.mrgn-bttm-sm").unwrap();
    let date_check_sel = Selector::parse("div.col-sm-4.text-right h5.mrgn-tp-0.mrgn-bttm-sm").unwrap();
    let p_sel       = Selector::parse("p").unwrap();

    //  – page loop
    let mut page = 1;
    'pages: loop {
        let url = format!(
            "https://search.open.canada.ca/grants/?page={}&sort=agreement_start_date+desc",
            page
        );
        let html = Client::new()
            .get(&url)
            .send().await?
            .text().await?;
        let document = Html::parse_document(&html);

        // if no items, stop
        let mut saw_any = false;

        for item in document.select(&item_sel) {
            saw_any = true;

            //extract the date for comparison
            let date_check = item
                .select(&date_check_sel)
                .next()
                .and_then(|div| div.text().next())
                .map(|s| s.trim().to_string())
                .unwrap_or_default()
                .trim()
                .to_string();
            
            // Try parsing, but *don’t* use `?` here:
            let parsed_date = match NaiveDate::parse_from_str(&date_check, "%b %e, %Y")
                // optional fallback to zero‑padded day:
                .or_else(|_| NaiveDate::parse_from_str(&date_check, "%b %d, %Y"))
            {
                Ok(d) => d,
                Err(e) => {
                    eprintln!("⚠️  Skipping record, could not parse date `{}`: {}", date_check, e);
                    continue;  // go to the next `item` in the loop
                }
            };

            // Now you have `parsed_date`, and you can turn it into a DateTime<Utc>:
            let dt = DateTime::<Utc>::from_utc(parsed_date.and_hms(0, 0, 0), Utc);

            // if older than one month, break out of everything
            if dt < one_month_ago {
                break 'pages;
            }

            // …now extract your other fields as before…
            let agreement = item
                .select(&info_sel)
                .next()
                .and_then(|div| div.select(&p_sel).next())
                .map(|p| p.inner_html().trim().to_string())
                .unwrap_or_default();
            let agreement_number = item
                .select(&info_sel)
                .filter(|div| div.text().any(|t| t.contains("Agreement Number")))
                .next()
                .and_then(|div| div.select(&p_sel).next())
                .map(|p| p.inner_html().trim().to_string())
                .unwrap_or_default();
            let description = item
                .select(&generic_sel)
                .filter(|div| div.text().any(|t| t.contains("Description")))
                .next()
                .and_then(|div| div.select(&p_sel).next().map(|p| p.inner_html()))
                .unwrap_or_default()
                .replace("Description", "")
                .trim()
                .to_string();
            let recipient = item
                .select(&generic_sel)
                .filter(|div| div.text().any(|t| t.contains("Organization")))
                .next()
                .map(|div| div.text().collect::<Vec<_>>().join(" "))
                .unwrap_or_default()
                .replace("Organization", "")
                .trim()
                .to_string();
            let recipient_public_name = item
                .select(&name_sel)
                .next()
                .and_then(|div| div.select(&p_sel).next().map(|p| p.inner_html()))
                .unwrap_or_default()
                .trim()
                .to_string();
            let date_range = item
                .select(&info_sel)
                .filter(|d| d.text().any(|t| t.contains("Duration")))
                .next()
                .map(|div| div.text().collect::<Vec<_>>().join(" "))
                .unwrap_or_default()
                .replace("Duration", "")
                .trim()
                .to_string();
            let price = item
                .select(&price_sel)
                .next()
                .map(|h4| h4.inner_html().trim().to_string())
                .unwrap_or_default();
            let location = item
                .select(&generic_sel)
                .filter(|div| div.text().any(|t| t.contains("Location")))
                .next()
                .map(|div| div.text().collect::<Vec<_>>().join(" "))
                .unwrap_or_default()
                .replace("Location", "")
                .trim()
                .to_string();

            // Print to console
            println!("Agreement:        {}", agreement);
            println!("Agreement Number: {}", agreement_number);
            println!("Date Range:             {}", date_range);
            println!("Date Agreed:      {}", date_check);
            println!("Description:      {}", description);
            println!("Recipient:        {}", recipient);
            println!("Recipient Public Name: {}", recipient_public_name);
            println!("Price:            {}", price);
            println!("Location:         {}", location);
            println!("──────────────────────────────────────────");

            // write to CSV
            wtr.write_record(&[
                &agreement,
                &agreement_number,
                &date_range,
                &date_check,
                &description,
                &recipient,
                &recipient_public_name,
                &price,
                &location,
            ])?;
        }

        if !saw_any {
            // no results on this page → stop
            break;
        }
        page += 1;
    }

    wtr.flush()?;
    Ok(())
}
