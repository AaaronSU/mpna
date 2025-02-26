#ifndef COOVECTOR_H
#define COOVECTOR_H

struct COOVector {
    int n;
    std::vector<int> indices;
    std::vector<double> values;
};

COOVector createDenseCOOVector(int n) {
    COOVector vec;
    vec.n = n;
    for (int i = 0; i < n; i++) {
        vec.indices.push_back(i);
        vec.values.push_back(1.0);
    }
    return vec;
}

COOVector csrMatrixVectorMultiply(const CSRMatrix<int, double>& A, const COOVector& x) {
    std::unordered_map<int, double> xMap;
    for (size_t i = 0; i < x.indices.size(); i++) {
        xMap[x.indices[i]] = x.values[i];
    }
    COOVector y;
    y.n = A.n;
    for (int i = 0; i < A.n; i++) {
        double sum = 0.0;
        for (int idx = A.row_ptr[i]; idx < A.row_ptr[i + 1]; idx++) {
            int col = A.col_idx[idx];
            double a_val = A.values[idx];
            auto it = xMap.find(col);
            if (it != xMap.end()) {
                sum += a_val * it->second;
            }
        }
        if (sum != 0.0) {
            y.indices.push_back(i);
            y.values.push_back(sum);
        }
    }
    return y;
}

#endif