def candidate_elimination(training_data):
    S = ['0'] * len(training_data[0][0])
    G = [['?'] * len(training_data[0][0])]
    
    for instance, label in training_data:
        if label == 'yes': 
            for i in range(len(S)):
                if S[i] == '0':
                    S[i] = instance[i]
                elif S[i] != instance[i]:
                    S[i] = '?'
            G = [g for g in G if all(g[i] == '?' or g[i] == instance[i] for i in range(len(g)))]
        elif label == 'no':
            new_G = []
            for g in G:
                for i in range(len(g)):
                    if g[i] == '?':
                        for val in set([example[0][i] for example in training_data]):
                            if val != instance[i]:
                                new_g = g[:i] + [val] + g[i+1:]
                                if any(all(new_g[j] == '?' or new_g[j] == S[j] or S[j] == '?' for j in range(len(new_g))) for new_g in new_G):
                                    continue
                                new_G.append(new_g)
            G = new_G
            S = [s for s in S if any(all(s[i] == '?' or s[i] == g[i] for i in range(len(s))) for g in G)]
    return S, G
training_data = [
    (['Sunny', 'Warm', 'Normal', 'Strong', 'Warm', 'Same'], 'yes'),
    (['Sunny', 'Warm', 'High', 'Strong', 'Warm', 'Same'], 'yes'),
    (['Rainy', 'Cold', 'High', 'Strong', 'Warm', 'Change'], 'no'),
    (['Sunny', 'Warm', 'High', 'Strong', 'Cool', 'Change'], 'yes')
]
S, G = candidate_elimination(training_data)
print("The most specific hypothesis (S) is:", S)
print("The most general hypotheses (G) are:", G)