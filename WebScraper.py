import urllib.request import urlopen
from bs4 import BeautifulSoup
import requests

# Here, we're just importing both Beautiful Soup and the Requests library
page_link = 'https://www.african-markets.com/en/stock-markets/bse'

page_response = urllib.request.urlopen(page_link)
htmlSource = page_response.read()
# page_response.close()
print(htmlSource)


# this is the url that we've already determined is safe and legal to scrape from.
# page_response = urlopen(page_link)
print(page_response.read())
# here, we fetch the content from the url, using the requests library
page_content = BeautifulSoup(page_response.read(), "lxml")
# we use the html parser to parse the url content and store it in a variable.
textContent = []
for i in range(0, 20):
    paragraphs = page_content.find_all("p")[i].text
    textContent.append(paragraphs)
print(textContent)
# In my use case, I want to store the speech data I mentioned earlier.  so in this example, I loop through the
# paragraphs, and push them into an array so that I can manipulate and do fun stuff with the data.
