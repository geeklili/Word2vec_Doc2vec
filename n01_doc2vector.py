import gensim
import jieba
from gensim.models.doc2vec import Doc2Vec, LabeledSentence


stop_word_li = [i[:-1] for i in open('./data/stop_list.txt', 'r', encoding='utf-8')]

TaggededDocument = gensim.models.doc2vec.TaggedDocument


def get_train():
    with open("./data/doc2vec_corpus.txt", 'r', encoding='utf-8') as doc:
        train_docs = []
        for i, text in enumerate(doc):
            word_list = [i for i in list(jieba.cut(text)) if i not in stop_word_li]
            length = len(word_list)
            word_list[length - 1] = word_list[length - 1].strip()
            document = TaggededDocument(word_list, tags=[i])
            train_docs.append(document)
        # print(train_docs)
        return train_docs


def train(x_train, size=200, epoch_num=1):
    doc_model = Doc2Vec(x_train, min_count=1, window=3, vector_size=size, sample=1e-3, negative=5, workers=4)
    doc_model.train(x_train, total_examples=doc_model.corpus_count, epochs=70)
    doc_model.save('./data/model_doc2vec')


def predict_doc():
    doc_model = Doc2Vec.load("./data/model_doc2vec")
    text_test = '收到实物与图片一模一样，实在很惊喜，包装的也很好，下次再来光顾哦！'
    text_cut = [i for i in list(jieba.cut(text_test)) if i not in stop_word_li]
    print(text_cut)
    inferred_vector_dm = doc_model.infer_vector(text_cut)
    sim_sentence = doc_model.docvecs.most_similar([inferred_vector_dm], topn=10)
    print(sim_sentence)
    sentence = [i[:-1] for i in open('./data/doc2vec_corpus.txt', 'r', encoding='utf-8')]
    for count, sim_val in sim_sentence:
        print(sim_val, count)
        print(sentence[count])


if __name__ == '__main__':
    train_x = get_train()
    train(train_x)
    predict_doc()
