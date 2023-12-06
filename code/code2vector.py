import os
import numpy as np
from scipy.io import savemat

import javalang
from gensim.models import Word2Vec

save_path = '../data/embedding/'

def extract_tokens_from_ast(node):
    tokens = []
    if isinstance(node, javalang.ast.Node):
        tokens.append(type(node).__name__)
        for child_node in node.children:
            tokens.extend(extract_tokens_from_ast(child_node))
    return tokens


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
                        java_file_path = '../data/archives/' + txtfile.split('.txt')[
                            0] + '/src/java' + '/' + java_path_parts + '.java'

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
                        vectors = np.array([model.wv[token] for token in tokens if token in model.wv])

                        # Average the vectors to get a single vector representation
                        if len(vectors) > 0:
                            vector = np.mean(vectors, axis=0)
                        else:
                            vector = np.zeros(model.vector_size)  # Or handle the empty case however you prefer

                        directory_name = txtfile.split('.txt')[0]
                        vector_directory = os.path.join(save_path, directory_name)
                        if not os.path.exists(vector_directory):
                            os.makedirs(vector_directory)

                        name = (txtfile.split('.txt')[0] + '_src_java_' + java_path_parts).replace('/', '_')
                        subdirectory = 'clean' if parts[1] == '0' else 'buggy'

                        # Construct the save path with the decided subdirectory
                        save_vector_path = os.path.join(vector_directory, subdirectory, name + '.npy')
                        directory_name = os.path.dirname(save_vector_path)
                        if not os.path.exists(directory_name):
                            os.makedirs(directory_name)

                        # Save the averaged vector as a .npy file
                        np.save(save_vector_path, vector)
