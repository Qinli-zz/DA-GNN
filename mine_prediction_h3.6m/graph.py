import numpy as np



class Graph():

    def __init__(self):

        self.get_edge()

        self.get_adjacency()



    def __str__(self):

        return self.A_j, self.A_b, self.A_s, self.A_t



    def get_edge(self):

        # joint

        self.joint_num_node = 20

        self_link = [(i, i) for i in range(self.joint_num_node)]

        neighbor_link_ = [(1, 2), (2, 3), (3, 4), (5, 6), (6, 7), (7, 8), (1, 9), (5, 9),

                          (9, 10), (10, 11), (11, 12), (10, 13), (13, 14), (14, 15),

                          (15, 16), (10, 17), (17, 18), (18, 19), (19, 20)]

        self.neighbor_link = [(i - 1, j - 1) for (i, j) in neighbor_link_]

        self.joint_edge = self_link + self.neighbor_link

        # bone

        self.bone_num_node = 19

        self_link = [(i, i) for i in range(self.bone_num_node)]

        neighbor_link_ = [(1, 2), (2, 3), (4, 5), (5, 6), (7, 9), (8, 9), (9, 10), (9, 12),

                          (9, 16), (10, 11), (12, 13), (13, 14), (14, 15), (16, 17), (17, 18),

                          (18, 19)]

        neighbor_link = [(i - 1, j - 1) for (i, j) in neighbor_link_]

        self.bone_edge = self_link + neighbor_link



    def get_adjacency(self):

        # joint

        A = np.zeros((self.joint_num_node, self.joint_num_node))

        for i, j in self.joint_edge:

            A[j, i] = 1

            A[i, j] = 1

        self.A_j = self.normalize_Ajb(A)

        # bone

        A = np.zeros((self.bone_num_node, self.bone_num_node))

        for i, j in self.bone_edge:

            A[j, i] = 1

            A[i, j] = 1

        self.A_b = self.normalize_Ajb(A)

        # source

        A_s = np.zeros((self.bone_num_node, self.joint_num_node))

        A_t = np.zeros((self.bone_num_node, self.joint_num_node))

        for i, link in enumerate(self.neighbor_link):

            A_s[i, link[0]] = 1

            A_t[i, link[1]] = 1



        A_all = A_s + A_t

        self.A_s = self.normalize_Ast(A_s, A_all)

        self.A_t = self.normalize_Ast(A_t, A_all)



    def normalize_Ajb(self, A):

        Dl = np.sum(A, 0)

        num_node = A.shape[0]

        Dn = np.zeros((num_node, num_node))

        for i in range(num_node):

            if Dl[i] > 0:

                Dn[i, i] = Dl[i] ** (-1)

        AD = np.dot(A, Dn)

        return AD



    def normalize_Ast(self, A, A_all):

        epsilon = 1e-6

        degree_mat = A_all.sum(0) * np.eye(len(A_all[-1]))

        # Since all nodes should have at least some edge, degree matrix is invertible

        inv_degree_mat = np.linalg.inv(degree_mat)

        A_t = (inv_degree_mat @ np.transpose(A)) + epsilon

        return np.transpose(A_t)