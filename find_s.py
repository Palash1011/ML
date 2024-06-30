def find_s(training_data):
    hypothesis = None
    for instance in training_data:
        if instance[-1] == "yes":  
            hypothesis = instance[:-1]
            break
    for instance in training_data:
        if instance[-1] == "yes":  
            for i in range(len(hypothesis)):
                if hypothesis[i] != instance[i]: 
                    hypothesis[i] = '?'

    return hypothesis
training_data = [
    ['Sunny', 'Warm', 'Normal', 'Strong', 'Warm', 'Same', 'yes'],
    ['Sunny', 'Warm', 'High', 'Strong', 'Warm', 'Same', 'yes'],
    ['Rainy', 'Cold', 'High', 'Strong', 'Warm', 'Change', 'no'],
    ['Sunny', 'Warm', 'High', 'Strong', 'Cool', 'Change', 'yes']
]
hypothesis = find_s(training_data)
print("Most specific hypothesis:", hypothesis)