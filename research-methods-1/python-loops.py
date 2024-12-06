# Loops and iterations

## break
# nums = [1,2,3,4,5]
# for num in nums:
#     if num == 3: # when the conditional is met
#         print("Found")
#         break # break statement breaks out of the loop
#     print(num)

## continue
# nums = [1,2,3,4,5]
# for num in nums:
#     if num == 3:
#         print("Found")
#         continue # Continues the loop even if it meets the conditional
#     print(num)

# # Nested loops - loop within a loop
# nums = [1,2,3,4,5]
# for num in nums: # for each number loop through each character in the string and iterate
#     for letter in 'abc':
#         print(num, letter)


# range 
for i in range(10):
    print(i)

# while loop - will go on forever  until the condition evaluates to false
x = 0
while x < 10: # condition
    if x == 5: # nested condition and break
        break
    print(x)
    x+=1 # increment x