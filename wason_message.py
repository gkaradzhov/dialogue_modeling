class WasonMessage:
    def __init__(self, origin, content, annotation_obj, identifier):
        self.identifier = identifier
        self.origin = origin
        self.content = content
        self.annotation = annotation_obj
        self.content_tokenised = []
        self.content_pos = []
        self.no_solution_text = ""

    def merge_annotations(self, external_annotation_object):
        new_ann_dict = {**self.annotation, **external_annotation_object}
        self.annotation = new_ann_dict


class WasonConversation:
    def __init__(self, identifier):
        self.raw_db_conversation = []
        self.wason_messages = []
        self.identifier = identifier

    def preprocess_everything(self, tagger):
        for item in self.wason_messages:
            doc = tagger(item.content)
            item.content_pos = [a.pos_ for a in doc]
            item.content_tokenised = [a.text for a in doc]

    def remove_solutions(self):
        initial_cards = self.get_initial_cards()

        for item in self.wason_messages:
            no_sol = [a if a not in initial_cards else '' for a in item.content_tokenised]
            item.no_solution_text = " ".join(no_sol)

    def get_initial_cards(self):
        cards = set()
        for rm in self.raw_db_conversation:
            if rm['message_type'] == "WASON_INITIAL":
                cards.update([l['value'] for l in rm['content']])
                break
        return cards

    def wason_messages_from_raw(self):
        for m in self.raw_db_conversation:
            if m['message_type'] == 'CHAT_MESSAGE':
                self.wason_messages.append(WasonMessage(origin=m['user_name'],
                                                        content=m['content'],
                                                        identifier=m['message_id'],
                                                        annotation_obj=None))

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
