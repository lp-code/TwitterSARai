import re


def split_into_tags_and_doc(tweet):
    """A typical VPD tweet contains hashtags, which are normally at the beginnig and
    often geographical (place or street names). We split these off and return the
    actual text separately."""
    tweet = tweet.replace("@Hordalandpoliti", "") # in old twitter account tweets
    tag_list = re.findall("#(\w+)", tweet)
    for tag in tag_list:
        tweet = tweet.replace("#" + tag, "")
    return "|".join(tag_list), re.sub("^[\W0-9]*", "", tweet)  # remove leading non-text
