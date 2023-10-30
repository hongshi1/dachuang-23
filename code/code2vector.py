import os
import numpy as np
from scipy.io import savemat

import javalang
from gensim.models import Word2Vec
#我写的一种读取将java文件转化为ast的方法,然后提取token,用于word2vec训练,转换为1

base_path = '/home/'
save_path = '../data/embedding_mat/'

def extract_tokens_from_ast(node):
    tokens = []
    if isinstance(node, javalang.ast.Node):
        tokens.append(type(node).__name__)
        for child_node in node.children:
            tokens.extend(extract_tokens_from_ast(child_node))
    return tokens


# This function takes a Java file path, parses it, and returns the tokens from its AST.
def generate_ast_from_java(java_file_path):
    with open(java_file_path, 'r') as file:
        code = file.read()
    tree = javalang.parse.parse(code)
    return extract_tokens_from_ast(tree)

if __name__ == '__main__':
    all_tokens = []

    # Check if pre-trained model exists
    if os.path.exists("java_code_word2vec.model"):
        model = Word2Vec.load("java_code_word2vec.model")
    else:
        # Gather tokens from all Java files referenced in the .txt files
        for txtfile in os.listdir('../data/txt'):
            if txtfile.endswith('.txt'):
                with open(os.path.join('../data/txt', txtfile), 'r') as file:
                    for line in file.readlines():
                        parts = line.split()
                        if len(parts) < 2:
                            continue
                        java_path_parts = parts[0].replace('.', '/')
                        java_file_path = '../data/archives/' + txtfile.split('.txt')[0] + '/src/java' + '/' + java_path_parts + '.java'

                        if os.path.exists(java_file_path):
                            tokens = generate_ast_from_java(java_file_path)
                            all_tokens.append(tokens)

        # Train the model
        model = Word2Vec(sentences=all_tokens, vector_size=100, window=5, min_count=1, workers=4)
        model.save("java_code_word2vec.model")

    # Convert tokens to vectors and save
    for txtfile in os.listdir('../data/txt'):
        if txtfile.endswith('.txt'):
            with open(os.path.join('../data/txt', txtfile), 'r') as file:
                for line in file.readlines():
                    parts = line.split()
                    if len(parts) < 2:
                        continue
                    java_path_parts = parts[0].replace('.', '/')
                    java_file_path = '../data/archives/' + txtfile.split('.txt')[
                        0] + '/src/java' + '/' + java_path_parts + '.java'

                    if os.path.exists(java_file_path):

                        tokens = generate_ast_from_java(java_file_path)
                        vectors = [model.wv[token] for token in tokens if token in model.wv]

                        directory_name = txtfile.split('.txt')[0]
                        mat_directory = os.path.join(save_path, directory_name)
                        if not os.path.exists(mat_directory):
                            os.makedirs(mat_directory)

                        name = (txtfile.split('.txt')[0] + '_src_java_' + java_path_parts).replace('/', '_')
                        subdirectory = 'clean' if parts[1] == '0' else 'buggy'

                        # Construct the save path with the decided subdirectory
                        save_mat_path = os.path.join(mat_directory, subdirectory, name + '.mat')
                        directory_name = os.path.dirname(save_mat_path)
                        if not os.path.exists(directory_name):
                            os.makedirs(directory_name)

                        # Save vectors to .mat file
                        savemat(save_mat_path, {'vectors': vectors})