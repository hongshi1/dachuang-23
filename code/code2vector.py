import os
import numpy as np
from scipy.io import savemat

import javalang
from gensim.models import Word2Vec

save_path = '../data/embedding/'
SELECTED_NODES = (javalang.tree.MethodInvocation, javalang.tree.SuperMethodInvocation,
                  javalang.tree.SuperConstructorInvocation,
                  javalang.tree.ClassCreator, javalang.tree.ArrayCreator,
                  javalang.tree.PackageDeclaration, javalang.tree.InterfaceDeclaration, javalang.tree.ClassDeclaration,
                  javalang.tree.ConstructorDeclaration, javalang.tree.MethodDeclaration,
                  javalang.tree.VariableDeclaration,
                  javalang.tree.FormalParameter, javalang.tree.IfStatement, javalang.tree.ForStatement,
                  javalang.tree.WhileStatement, javalang.tree.DoStatement, javalang.tree.AssertStatement,
                  javalang.tree.BreakStatement, javalang.tree.ContinueStatement, javalang.tree.ReturnStatement,
                  javalang.tree.ThrowStatement, javalang.tree.TryStatement, javalang.tree.SynchronizedStatement,
                  javalang.tree.SwitchStatement, javalang.tree.BlockStatement, javalang.tree.CatchClauseParameter,
                  javalang.tree.TryResource, javalang.tree.CatchClause, javalang.tree.SwitchStatementCase,
                  javalang.tree.ForControl, javalang.tree.EnhancedForControl, javalang.tree.BasicType,
                  javalang.tree.MemberReference, javalang.tree.ReferenceType, javalang.tree.SuperMemberReference,
                  javalang.tree.StatementExpression)  # https://github.com/c2nes/javalang/blob/master/javalang/tree.py

NODES_STRING = ['javalang.tree.MethodInvocation', 'javalang.tree.SuperMethodInvocation',
                'javalang.tree.SuperConstructorInvocation',
                'javalang.tree.ClassCreator', 'javalang.tree.ArrayCreator',
                'javalang.tree.PackageDeclaration', 'javalang.tree.InterfaceDeclaration',
                'javalang.tree.ClassDeclaration',
                'javalang.tree.ConstructorDeclaration', 'javalang.tree.MethodDeclaration',
                'javalang.tree.VariableDeclaration',
                'javalang.tree.FormalParameter', 'javalang.tree.IfStatement', 'javalang.tree.ForStatement',
                'javalang.tree.WhileStatement', 'javalang.tree.DoStatement', 'javalang.tree.AssertStatement',
                'javalang.tree.BreakStatement', 'javalang.tree.ContinueStatement', 'javalang.tree.ReturnStatement',
                'javalang.tree.ThrowStatement', 'javalang.tree.TryStatement', 'javalang.tree.SynchronizedStatement',
                'javalang.tree.SwitchStatement', 'javalang.tree.BlockStatement', 'javalang.tree.CatchClauseParameter',
                'javalang.tree.TryResource', 'javalang.tree.CatchClause', 'javalang.tree.SwitchStatementCase',
                'javalang.tree.ForControl', 'javalang.tree.EnhancedForControl', 'javalang.tree.BasicType',
                'javalang.tree.MemberReference', 'javalang.tree.ReferenceType', 'javalang.tree.SuperMemberReference',
                'javalang.tree.StatementExpression']


def index_type(node, SELECTED_NODES):
    i = 0
    for key in SELECTED_NODES:
        i += 1
        if isinstance(node, key):
            return i  # Forced end of for loop
    return 0


def tokenize(node, SELECTED_NODES):
    i = 0
    for key in SELECTED_NODES:  # Traverse each node in SELECTED_NODES
        if isinstance(node, key):
            str = NODES_STRING[i]
            temp = str.split('.')
            return temp[-1]  # Return the last string and force end loop
            # return i  # Retrun the index
        i += 1
    return 'UKN'


def get_tokens(src_code_path):
    temp = ''
    if os.path.exists(src_code_path):  # If the source code of i-th module (i.e., a Java file) exists
        file = open(src_code_path, 'r')
        txt_java_code = file.read()
        tree_AST = javalang.parse.parse(txt_java_code)  # Obtain AST of i-th module
        for path, node in tree_AST:  # Traverse each node in AST
            mapping_token = tokenize(node, SELECTED_NODES)  # self-defined function 'tokenize'
            temp = temp + ' ' + mapping_token
    else:
        temp = None
    return temp


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
                            tokens = get_tokens(java_file_path)
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
                        # tokens = generate_ast_from_java(java_file_path)
                        tokens = get_tokens(java_file_path)
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
