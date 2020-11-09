import spacy


class WasonMessage:
    def __init__(self, origin, content, annotation_obj):
        self.origin = origin
        self.content = content
        self.annotation = annotation_obj
        self.content_tokenised = []
        self.content_pos = []

    def tokenise_content(self):
        # TODO: Add better tokeniser
        self.content_tokenised = self.content.split()


class WasonConversation:
    def __init__(self, identifier):
        self._raw_db_conversation = []
        self.wason_messages = []
        self.identifier = identifier

    def preprocess_everything(self, tagger):
        for item in self.wason_messages:
            doc = tagger(item.content)
            item.content_pos = [a.pos_ for a in doc]
            item.content_tokenised = [a.text for a in doc]

        

    def to_street_crowd_format(self):
        data = []
        for count, item in enumerate(self.wason_messages):
            data.append((item.origin, " ".join(item.content_tokenised), " ".join(item.content_pos), count))

        return data
