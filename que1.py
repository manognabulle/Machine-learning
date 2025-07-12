s= input("enter string:")
vowels="aeiouAEIOU"
v_count=0
c_count=0

for ch in s:
    if ch.isalpha():
        if ch in vowels:
            v_count+=1
        else:
            c_count+=1

print("vowels:",v_count)
print("Consonants:",c_count)
