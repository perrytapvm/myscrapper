import requests
from bs4 import BeautifulSoup
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

def scrape_news(url):
    # Получение HTML-кода страницы
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    # Извлечение заголовков и текстов новостей
    headlines = []
    articles = []
    for article in soup.find_all('article'):
        headline = article.h3.text.strip()
        text = article.find('div', class_='js-article-inner').text.strip()
        headlines.append(headline)
        articles.append(text)

    # Создание DataFrame для хранения данных
    news_df = pd.DataFrame({'Headline': headlines, 'Article': articles})

    return news_df

def analyze_and_visualize(news_df):
    # Векторизация текстов новостей
    vectorizer = CountVectorizer(stop_words='english', max_features=1000)
    X = vectorizer.fit_transform(news_df['Article'])

    # Применение Latent Dirichlet Allocation для выделения тем
    lda = LatentDirichletAllocation(n_components=5, random_state=42)
    lda.fit(X)

    # Визуализация результатов
    feature_names = vectorizer.get_feature_names_out()
    for topic_idx, topic in enumerate(lda.components_):
        top_words_idx = topic.argsort()[:-10 - 1:-1]
        top_words = [feature_names[i] for i in top_words_idx]
        print(f"Topic #{topic_idx + 1}: {', '.join(top_words)}")

    # Гистограмма распределения новостей по темам
    topic_distribution = lda.transform(X)
    topic_labels = [f"Topic {i+1}" for i in range(lda.n_components)]
    plt.bar(topic_labels, topic_distribution.sum(axis=0))
    plt.xlabel('Topics')
    plt.ylabel('Number of Articles')
    plt.title('Distribution of News Articles Across Topics')
    plt.show()

if __name__ == "__main__":
    bbc_news_url = 'https://www.bbc.com/news'
    news_data = scrape_news(bbc_news_url)
    analyze_and_visualize(news_data)
