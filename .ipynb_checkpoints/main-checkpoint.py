import pandas as pd 
from sklearn.model_selection import train_test_split
"""
Define categories that we will use to sample the data, putting emphasis on diversity
"""

categories = [
"Beauty/Fragrance/Women",
"Beauty/Skin Care/Face",
"Men/Tops/T-shirts",
"Electronics/Video Games & Consoles/Games",
"Women/Dresses/Above Knee, Mini",
"Women/Women's Handbags/Shoulder Bag",
"Women/Athletic Apparel/Pants, Tights, Leggings",
"Women/Shoes/Boots",
"Women/Underwear/Bras",
'"Electronics/Cell Phones & Accessories/Cases, Covers & Skins"',
"Women/Jewelry/Necklaces",
"Women/Tops & Blouses/T-Shirts",
"Kids/Toys/Dolls & Accessories", 
"Men/Shoes/Athletic",
"Home/Home Décor/Home Décor Accents",
"Kids/Toys/Action Figures & Statues",
"Electronics/Media/DVD",
"Other/Daily & Travel items/Personal Care",
"Vintage & Collectibles/Toy/Action Figure",
"Home/Kitchen & Dining/Dining & Entertaining",
"Other/Office supplies/Shipping Supplies",
"Handmade/Paper Goods/Sticker",
"Vintage & Collectibles/Trading Cards/Sports"
]


"""
Create training and validation split,
Add into 100k sample which is split 70/15
"""

data = pd.read_csv("train.csv", sep='\t')

filtered = data[
    (data['item_description'] != 'No description yet') &
    (data['category_name'].notna())
].copy()
filtered = filtered[filtered['category_name'].isin(categories)]

# Build balanced 100k pool from categories
category_samples = []
per_category = 100000 // len(categories)  # ~4347 per category
for category in categories:
    cat_data = filtered[filtered['category_name'] == category]
    cat_sample = cat_data.sample(min(len(cat_data), per_category), random_state=42)
    category_samples.append(cat_sample)

pool = pd.concat(category_samples, ignore_index=True)
print(f"Total pool size: {len(pool)}")

# Split pool into 85k train and 15k validation
train_set, val_set = train_test_split(pool, test_size=0.15, random_state=42)
print(f"Train size: {len(train_set)}")
print(f"Validation size: {len(val_set)}")

# train_set.to_csv('train_sample.csv', index=False)
# val_set.to_csv('validation_sample.csv', index=False)



"""
Lastly we create approximately a 15k test set to complete the 70/15/15 split 
"""

# Test set from test.csv
testing_data = pd.read_csv("test.csv", sep='\t')

filtered_test = testing_data[
    (testing_data['item_description'] != 'No description yet') &
    (testing_data['category_name'].notna())
].copy()
filtered_test = filtered_test[filtered_test['category_name'].isin(categories)]

test_samples = []
for category in categories:
    cat_data = filtered_test[filtered_test['category_name'] == category]
    cat_sample = cat_data.sample(min(len(cat_data), 650), random_state=42)
    test_samples.append(cat_sample)

test_set = pd.concat(test_samples, ignore_index=True)

# Remove any descriptions that appear in training data
test_clean = test_set[~test_set['item_description'].isin(train_set['item_description'])]
print(f"Test size after removing overlaps: {len(test_clean)}")
# test_clean.to_csv('test_sample15k_clean.csv', index=False)

train_set['category_name'].value_counts().to_csv('train_category_distribution.csv')
val_set['category_name'].value_counts().to_csv('validation_category_distribution.csv')
test_clean['category_name'].value_counts().to_csv('test_category_distribution.csv')

