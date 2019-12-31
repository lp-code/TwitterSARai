import re


def split_into_tags_and_doc(tweet):
    tweet = tweet.replace("@Hordalandpoliti", "") # in old twitter account tweets
    tag_list = re.findall("#(\w+)", tweet)
    for tag in tag_list:
        tweet = tweet.replace("#" + tag, "")
    return "|".join(tag_list), re.sub("^[\W0-9]*", "", tweet)  # remove leading non-text
