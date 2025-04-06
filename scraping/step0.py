from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
import json
import time

def extract_table_data(table):
    rows = table.find_elements(By.XPATH, ".//tr")[1:]  # skip header
    data = []

    for row in rows:
        try:
            name_elem = row.find_element(By.XPATH, ".//td[1]/a")
            name = name_elem.text.strip()
            link = name_elem.get_attribute("href")

            remote_testing = "Yes" if "catalogue__circle -yes" in row.find_element(By.XPATH, ".//td[2]").get_attribute("outerHTML") else "No"
            adaptive_irt = "Yes" if "catalogue__circle -yes" in row.find_element(By.XPATH, ".//td[3]").get_attribute("outerHTML") else "No"

            test_types = [el.text.strip() for el in row.find_elements(By.XPATH, ".//td[4]//span[@class='product-catalogue__key']")]

            data.append({
                "name": name,
                "link": link,
                "remote_testing": remote_testing,
                "adaptive_irt": adaptive_irt,
                "test_types": test_types
            })
        except Exception as e:
            print(f"⚠️ Skipping row due to error: {e}")
            continue
    return data

def scrape_prepackaged(driver, page_count=12):
    print("📘 Scraping Pre-packaged Job Solutions (12 pages)")
    all_data = []
    for i in range(page_count):
        url = f"https://www.shl.com/solutions/products/product-catalog/?start={i * 12}&type=2"
        print(f"🔍 Page {i+1}: {url}")
        driver.get(url)
        time.sleep(2)
        try:
            table = driver.find_element(By.XPATH, "(//div[@class='custom__table-responsive']/table)[1]")
            data = extract_table_data(table)
            all_data.extend(data)
        except Exception as e:
            print(f"❌ Error on page {i+1}: {e}")
    return all_data

def scrape_individual(driver, page_count=32):
    print("📗 Scraping Individual Test Solutions (32 pages)")
    all_data = []
    for i in range(page_count):
        url = f"https://www.shl.com/solutions/products/product-catalog/?start={i * 12}&type=1"
        print(f"🔍 Page {i+1}: {url}")
        driver.get(url)
        time.sleep(2)
        try:
            if i == 0:
                table = driver.find_element(By.XPATH, "(//div[@class='custom__table-responsive']/table)[2]")
            else:
                table = driver.find_element(By.XPATH, "(//div[@class='custom__table-responsive']/table)[1]")
            data = extract_table_data(table)
            all_data.extend(data)
        except Exception as e:
            print(f"❌ Error on page {i+1}: {e}")
    return all_data

def main():
    options = Options()
    # options.add_argument("--headless")  # uncomment to run headless
    driver = webdriver.Chrome(options=options)

    prepackaged_data = scrape_prepackaged(driver)
    individual_data = scrape_individual(driver)

    driver.quit()

    result = {
        "pre_packaged_solutions": prepackaged_data,
        "individual_test_solutions": individual_data
    }

    with open("shl_product_links_final.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print("✅ Scraping complete! Data saved to 'shl_product_links_final.json'")

if __name__ == "__main__":
    main()
