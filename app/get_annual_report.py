from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

driver = webdriver.Safari()

driver.get("http://www.sse.com.cn/disclosure/listedinfo/regular/")

driver.maximize_window()

driver.find_element_by_id("inputCode").clear()
driver.find_element_by_id("inputCode").send_keys("600000")

js="document.getElementById('start_date').removeAttribute('readonly')"
driver.execute_script(js)
driver.find_element_by_id("start_date").clear()
driver.find_element_by_id("start_date").send_keys("2020-04-03")

js="document.getElementById('end_date').removeAttribute('readonly')"
driver.execute_script(js)
driver.find_element_by_id("end_date").clear()
driver.find_element_by_id("end_date").send_keys("2021-04-03")

js='document.getElementById("btnQuery").click()'
driver.execute_script(js)

table_search = driver.find_element_by_css_selector("[class='table search_']")

table_header = table_search.find_elements_by_tag_name("tr")[0].find_elements_by_tag_name("th")
columns = []
for th in table_header:
    columns.append(th.text)


table_row = table_search.find_elements_by_tag_name("tr")[1:]

final_ret = []
for tr in table_row:
    td = tr.find_elements_by_tag_name("td")
    ret = []
    for col in td:
        ret.append(col.text)
    link = td.find_element_by_xpath("//*[@href]").get_attribute('href')
    print(link)
    ret.append(link)
    final_ret.append(ret)

