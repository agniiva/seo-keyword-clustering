# SEO Topic Cluster Creation

This is my process for creating content clusters, topical maps—whatever you want to call them. Ultimately, this is how I plan content.

[Check out more of my work at CashCowLabs.io](https://cashcowlabs.io)

---

## The Spark That Ignited the Journey

A few months back, I found myself wrestling with the complexities of SEO, particularly with the concept of topical authority. As search engines evolved, it became clear that **organizing content around coherent topics** was becoming crucial. I needed a reliable system for creating topical maps or clustering keywords—a pressing need in my day-to-day work.

I pondered over various approaches, diving deep into algorithms and existing tools. But nothing seemed to fit just right. Then, a thought struck me: **"The answer is always on the first page of Google."** I can't recall whether it was Kyle Roof or Matt Diggity who said that, but it resonated with me.

I realized that **Google itself was already grouping keywords for us**. If a single page ranks for multiple keywords, then those keywords are related in Google's eyes. So, why not use Google's own SERPs to inform our keyword clustering?

## Using Google's Wisdom at play

With this hypothesis, I set out to build a tool that could use Google's understanding of keyword relationships. The idea was simple:

1. **Gather a comprehensive list of keywords** around a primary topic. (Using various tools like Ahrefs/Semrush - standard procedure)
2. **Scrape the SERPs** for each keyword to find the top-ranking URLs.
3. **Group keywords by overlapping URLs**, effectively letting Google show us which keywords belong together.

## So the Process

### Gathering Keywords

I started by casting a wide net. Let's say the primary topic is **coffee**. I used keyword research tools [Ahrefs] to pull in as many related keywords as possible—digging into subtopics like brewing methods, bean types, roasting techniques, and more.

To keep things manageable, I applied a few filters:

- Monthly Search Volume > 0
- Language: English
- Long-tail Keyowords

I wasn't too picky at this stage. The goal was to collect a broad dataset to work with.
<img width="1695" alt="image" src="https://github.com/user-attachments/assets/db5a5113-5c80-40ad-bb26-1f0f3c452429">


### Scraping the SERPs (But Be Careful!)

Next came the scraping part. I needed to retrieve the top search results for each keyword. While the noble way is to use APIs like the [Google Custom Search API](https://developers.google.com/custom-search/v1/overview), it can get pricey and comes with limitations.

Instead, I opted for **Serper.dev's API**, which offers a more affordable solution—around **$1 per 1,000 keywords**, compared to other tools that charge upwards of $5 for the same amount. [You also get free 2500 scraping with multi-batch processing, so it was super awesome]

*Note: Scraping Google's search results without permission is against their TOS.*

<img width="1702" alt="image" src="https://github.com/user-attachments/assets/7ba7653b-b92b-458e-87cd-af582832bff8">


### Clustering Keywords by URL Overlaps

With the SERP data in hand, I began grouping keywords based on overlapping URLs. If two keywords had several common top-ranking URLs, it indicated that Google considered them closely related.

I utilized an **agglomerative clustering algorithm** for this. It's a hierarchical method that starts by treating each keyword as its own cluster and then merges them based on similarity—in this case, the number of overlapping URLs.

<img width="1685" alt="image" src="https://github.com/user-attachments/assets/355509dd-9004-4ebb-b690-83f6c0a67c92">



### Adding Intent Classification

To enrich the clusters further, I incorporated **intent classification** using **Sentence Transformers**. This step involved analyzing the titles of the top-ranking pages to determine whether the user's intent was informational, commercial, or navigational.

While this added depth to the clusters, it also increased processing time. The Sentence Transformer model is powerful but resource-intensive. I found it worth the wait, though, as understanding intent is crucial for crafting content that truly meets user needs.

## Overcoming Challenges Along the Way

### Balancing Depth and Efficiency

One of the biggest hurdles was balancing the thoroughness of the clustering with the time and resources it required. Running intent classification on large datasets can be time-consuming.

To mitigate this, I:

- Optimized the code for better performance.
- Allowed users to adjust parameters like the minimum number of overlapping URLs required for clustering.
- Made the intent classification optional, so users could choose based on their priorities.

### Navigating the Ethical Landscape

Scraping data always comes with ethical considerations. I was cautious to use APIs that respect the terms of service and privacy policies. While tools exist to scrape data directly from SERPs, I **do not recommend or endorse unauthorized scraping**.

## Building the Tool: A Technical Overview

While I won't dive deep into the code here, I used a combination of powerful Python libraries:

- **Streamlit** for the user interface, making the tool interactive and accessible.
- **SQLite** for efficient data storage and retrieval.
- **NetworkX** for building and analyzing graphs based on URL overlaps.
- **Transformers** library for intent classification with models like `facebook/bart-large-mnli`.

---

## How You Can Use It

I've made the tool available on GitHub for anyone interested.

[![How to use the tool](https://img.youtube.com/vi/ib8KmRKk7l4/0.jpg)](https://www.youtube.com/watch?v=ib8KmRKk7l4)

### Requirements

- Python 3.7 or higher
- API key from Serper.dev

### Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/yourusername/seo-keyword-clustering-tool.git
   cd seo-keyword-clustering-tool
   ```

2. **Install the Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

### Running the Tool

```bash
streamlit run app.py
```

### Steps to Follow

1. **Create or Select a Project**

   - Launch the app and either select an existing project or create a new one.

2. **Upload Keywords**

   - Upload a CSV file containing your keywords and their search volumes.
   - Make sure the file has `keyword` and `volume` columns. 

3. **Set Parameters and Scrape SERPs**

   - Enter your Serper.dev API key.
   - Set the minimum number of overlapping URLs for clustering.
   - Click on "Save Data and Start Scraping" to begin.

4. **Clustering**

   - Navigate to the Clustering page.
   - Adjust the overlap threshold if needed.
   - Start the clustering process.
   - **Optionally**, download the clusters as a CSV file.

5. **Intent Classification**

   - On the Results page, you can start the intent classification.
   - Be patient; this process can take some time due to the Sentence Transformer.

6. **Explore and Export Results**

   - View your clusters along with their dominant intent.
   - Download the final results as a CSV file.

---

Creating this tool was both challenging and rewarding. It allowed me to deepen my understanding of SEO and data analysis. More importantly, it gave me a practical solution that saves time and enhances the quality of my team's content planning.


## Why This Matters

At [CashCowLabs.io](https://cashcowlabs.io), my goal is to drive rapid SEO growth for clients. This tool helps streamlining the content planning process for my team and helping create more effective content clusters for my clients.

By leveraging Google's own data and sophisticated algorithms, we can stay ahead in the ever-evolving SEO landscape.

---

*Feel free to reach out if you have any projects, questions or insights. Happy clustering bois!*
