from nltk.tokenize import TweetTokenizer


class WasonMessage:
    def __init__(self, origin, content, annotation_obj, identifier, type='MESSAGE'):
        self.identifier = identifier
        self.origin = origin
        self.content = content
        self.annotation = annotation_obj
        self.content_tokenised = []
        self.content_pos = []
        self.clean_text = ""
        self.type = type

    def merge_annotations(self, external_annotation_object):
        new_ann_dict = {**self.annotation, **external_annotation_object}
        self.annotation = new_ann_dict


class WasonConversation:
    def __init__(self, identifier):
        self.raw_db_conversation = []
        self.wason_messages = []
        self.identifier = identifier
        self.tknzr = TweetTokenizer()

    def preprocess_everything(self, tagger):
        for item in self.wason_messages:
            doc = tagger(item.content)
            item.content_pos = [a.pos_ for a in doc]
            item.content_tokenised = self.tknzr.tokenize(item.content)

    def clean_special_tokens(self):
        initial_cards = self.get_initial_cards()
        users = self.get_users()
        users_upper = [u.upper() for u in users if u != 'SYSTEM']
        for item in self.wason_messages:
            clean_tokens = []
            for token in item.content_tokenised:
                if token.upper() in initial_cards:
                    clean_tokens.append('<CARD>')
                elif token.upper().replace('@', '') in users_upper:
                    clean_tokens.append('<MENTION>')
                else:
                    clean_tokens.append(token)

            item.clean_text = " ".join(clean_tokens)

    def get_initial_cards(self):
        cards = set()
        for rm in self.raw_db_conversation:
            if rm['message_type'] == "WASON_INITIAL":
                cards.update([l['value'] for l in rm['content']])
                break
        return cards

    def get_users(self):
        users = set()
        for item in self.wason_messages:
            users.add(item.origin)
        return users

    def wason_messages_from_raw(self):
        self.wason_messages = []
        for m in self.raw_db_conversation:
            if m['message_type'] == 'CHAT_MESSAGE':
                self.wason_messages.append(WasonMessage(origin=m['user_name'],
                                                        content=m['content'],
                                                        identifier=m['message_id'],
                                                        annotation_obj={}))

    def to_street_crowd_format(self):
        data = []
        for count, item in enumerate(self.wason_messages):
            data.append((item.origin, " ".join(item.content_tokenised), " ".join(item.content_pos), count))

        return data

    def get_wason_from_raw(self, raw_message):
        processed_message = [m for m in self.wason_messages if m.identifier == raw_message['message_id']]
        if len(processed_message) == 0:
            return None
        return processed_message[0]

    def merge_all_annotations(self, external_conversation):
        for internal, external in zip(self.wason_messages, external_conversation.wason_messages):
            if internal.identifier == external.identifier:
                internal.merge_annotations(external.annotation)
            else:
                print("Internal != External: {} {}".format(internal.identifier, external.identifier))
