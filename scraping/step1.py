import json
import time
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By

# Load product link data
with open("shl_product_links_final.json", "r", encoding="utf-8") as f:
    product_data = json.load(f)

# Test type meaning map
TEST_TYPE_MAPPING = {
    "A": "Ability & Aptitude",
    "B": "Biodata & Situational Judgement",
    "C": "Competencies",     
    "D": "Development & 360",
    "E": "Assessment Exercises",
    "K": "Knowledge & Skills",
    "P": "Personality & Behavior",
    "S": "Simulations"
}

# Setup Selenium WebDriver
options = Options()
options.add_argument("--start-maximized")
# Uncomment below to run headless
# options.add_argument("--headless")
driver = webdriver.Chrome(options=options)

# Helper function to extract values
def safe_xpath(xpath):
    try:
        return driver.find_element(By.XPATH, xpath).text.strip()
    except:
        return ""

def get_downloads():
    downloads = []
    elements = driver.find_elements(By.XPATH, "//ul[@class='product-catalogue__downloads']//li")
    for li in elements:
        try:
            a = li.find_element(By.TAG_NAME, "a")
            lang = li.find_element(By.CLASS_NAME, "product-catalogue__download-language")
            downloads.append({
                "title": a.text.strip(),
                "url": a.get_attribute("href"),
                "language": lang.text.strip()
            })
        except:
            continue
    return downloads

# Map letter codes to full meaning
def map_test_types(codes):
    return [TEST_TYPE_MAPPING.get(code, code) for code in codes]

# Main page details
def extract_product_details(url):
    driver.get(url)
    time.sleep(2)

    return {
        "title": safe_xpath("//h1"),
        "description": safe_xpath("//div[contains(@class, 'product-catalogue-training-calendar__row')][h4='Description']/p"),
        "job_levels": safe_xpath("//div[contains(@class, 'product-catalogue-training-calendar__row')][h4='Job levels']/p").rstrip(","),
        "languages": safe_xpath("//div[contains(@class, 'product-catalogue-training-calendar__row')][h4='Languages']/p").rstrip(","),
        "assessment_length": safe_xpath("//div[contains(@class, 'product-catalogue-training-calendar__row')][h4='Assessment length']/p"),
        "downloads": get_downloads()
    }

# Result container
product_details = {
    "pre_packaged_solutions": [],
    "individual_test_solutions": []
}

# Loop through and scrape
for category in product_data:
    for idx, product in enumerate(product_data[category], 1):
        print(f"[{idx}/{len(product_data[category])}] Scraping: {product['link']}")
        details = extract_product_details(product["link"])

        # Map test types before storing
        if "test_types" in product:
            product["test_types"] = map_test_types(product["test_types"])

        product.update(details)
        product_details[category].append(product)

driver.quit()

# Save final JSON
with open("shl_product_details_full.json", "w", encoding="utf-8") as f:
    json.dump(product_details, f, indent=2, ensure_ascii=False)

print("✅ All product details saved to 'shl_product_details_full.json'")
