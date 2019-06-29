import pandas as pd
import os
import jieba
from jieba import analyse
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

# 分类提取，首先读取这个文件夹中的所有txt文档，取每篇文章关键词，用空格连接成字符串
# 然后写入一个新的txt文档，存入列表（也可以写成一篇新文档）
# 将文章内容（Tfidf方法）和目标table（字典格式转换DataFrame）分别转化为DataFrame格式
# 14个类别中分别挑出30%的数据作为测试集，剩下的70%作为训练集数据；
# 利用贝叶斯模型训练，得出预测值
# 利用分类模型评估api，查看模型效果

# 数据所在路径
diff_kind_dir_path = 'E:\\python\\数据\\THUCNews'
# 设定自定义停用词表
analyse.set_stop_words("E:\\python\\数据\\stopwords-master\\哈工大停用词表.txt")

# 读取文本数据，返回字典形式数据{'体育'：[...], '娱乐':[...],...
def get_text():
    '''获取文档信息，每篇文章提取关键词
    return： kind_dic, diff_kind_dir_list
    '''
    # 将不同种类的新闻数据分组提取出来
    diff_kind_dir_list = os.listdir(diff_kind_dir_path)
    # diff_kind_dir_list为总的种类名
    print(diff_kind_dir_list)
    diff_kind_dir_path_list = [os.path.join(diff_kind_dir_path, dir_name) for dir_name in diff_kind_dir_list]
    print(diff_kind_dir_path_list)
    # 分别提取不同种类分组中的文章地址
    kind_dic = {} # 字典的键为种类名，value为对应类名的文章的地址
    # kind_list = [] # 每篇文章对应的种类名
    for diff_kind_dir in diff_kind_dir_path_list:
        kind_name = diff_kind_dir.split('\\')[-1]
        # kind_list.append(kind_name)
        file_list = os.listdir(diff_kind_dir)
        file_path_list = [os.path.join(diff_kind_dir, file_name) for file_name in file_list]
        kind_dic[kind_name] = file_path_list
    # print(kind_list)
    return kind_dic, diff_kind_dir_list

# 提取每篇文章中的关键词，同一类别合并所有关键词，以及出现频数
def key_words(kind_dic, kind_list):
    """
    提取每篇文章的关键词，经过停用词过滤后组成新的字符串，形成列表，同时对应生成每篇文章类型列表
    :param kind_dic: 
    :param kind_list: 
    :return: content_list, key_list
    """
    content_list = []
    key_list = []
    # 关键词抽取
    words_extractor = analyse.extract_tags
    for kind_name in kind_list:
        i = 0
        for file_name in kind_dic[kind_name]:
            # 直接用read读取文件内容即可
            with open(file_name, encoding='utf8') as f:
                content = f.read()
            important_words = words_extractor(content, topK=20) # 确定提取词数目
            # print(important_words)

            # # 将提取词组成新文件内容（词*词频） 此时计算准确率有所下降
            # words_list = jieba.cut(content)
            # new_content = []
            # for word in words_list:
            #     if word in important_words:
            #         new_content.append(word)
            # content_extrac = ' '.join(new_content)
            content_extrac = ' '.join(important_words)
            print(content_extrac)
            content_list.append(content_extrac)
            key_list.append(kind_name)

            # 防止数据量过大，内存溢出，每类文章取1000个
            if i >= 1000:
                print(kind_name)
                break
            i += 1
    return content_list, key_list

def nbc(content_list, key_list, kind_list):
    """
    将文件内容以及目标值列表读成DataFrame格式，划分数据集之后进行贝叶斯训练，查看模型预测效果
    :param content_list: 
    :param key_list: 
    :param kind_number: 
    :return: None
    """
    # 先实例化训练模型
    tf = TfidfVectorizer()
    # Tfidf提取每篇文章关键词以及词频，组成array列表形式
    value_array = tf.fit_transform(content_list).toarray()
    # 将提取的单词列表以及目标数据转化为DataFrame格式
    value_df = pd.DataFrame(value_array, columns=tf.get_feature_names())
    key_df = pd.DataFrame({'label':key_list})
    # 划分训练集与测试集
    train_value, test_value, train_label, test_label = train_test_split(value_df, key_df, test_size=0.3)
    # 实例化贝叶斯模型
    mlt = MultinomialNB(alpha=1.0) # alpha为拉普拉斯平滑系数
    mlt.fit(train_value, train_label)
    # 贝叶斯预测
    predict_label = mlt.predict(test_value)
    # 模型评估
    bayes_report = classification_report(test_label, predict_label, target_names=kind_list)
    print(bayes_report)


if __name__ == '__main__':
    kind_dic, kind_list = get_text()
    content_list, key_list = key_words(kind_dic, kind_list)
    nbc(content_list, key_list, kind_list)
