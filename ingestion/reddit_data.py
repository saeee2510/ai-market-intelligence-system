import praw
from config import REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USER_AGENT

def fetch_reddit():
    reddit = praw.Reddit(
        client_id=REDDIT_CLIENT_ID,
        client_secret=REDDIT_CLIENT_SECRET,
        user_agent=REDDIT_USER_AGENT
    )

    posts = []
    for post in reddit.subreddit("wallstreetbets").hot(limit=50):
        posts.append(post.title)

    return posts

if __name__ == "__main__":
    print(fetch_reddit()[:10])