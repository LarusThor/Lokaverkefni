import pandas as pd 

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
Create training split,
Add into 100k sample
"""

data = pd.read_csv("train.csv", sep='\t')


filtered = data[
    (data['item_description'] != 'No description yet') &
    (data['category_name'].notna())
].copy()

filtered = filtered[filtered['category_name'].isin(categories)]


# Sample each category separately and concatenate
samples = []
for category in categories:
    cat_data = filtered[filtered['category_name'] == category]
    cat_sample = cat_data.sample(min(len(cat_data), 4500), random_state=42)
    samples.append(cat_sample)

sample = pd.concat(samples, ignore_index=True)
train_sample = sample.sample(min(len(sample), 100000), random_state=42)

"""
Create test split,
Add into 20k sample
"""

testing_data = pd.read_csv("test.csv", sep='\t')

#(testing_data['category_name'].value_counts()).to_csv('test_set_category_distribution.csv')

filtered = testing_data[
    (testing_data['item_description'] != 'No description yet') &
    (testing_data['category_name'].notna())
].copy()

filtered = filtered[filtered['category_name'].isin(categories)]

samples = []
for category in categories:
    cat_data = filtered[filtered['category_name'] == category]
    cat_sample = cat_data.sample(min(len(cat_data), 900), random_state=42)
    samples.append(cat_sample)

sample = pd.concat(samples, ignore_index=True)
test_sample = sample.sample(min(len(sample), 20000), random_state=42)


#sample.to_csv('test_sample10k.csv', index=False)

#(sample['category_name'].value_counts()).to_csv('adjusted_test_set_category_distribution.csv')

"""
Clean test sample and delete instances 
that contain duplicate item descriptions
"""

test_clean = test_sample[~test_sample['item_description'].isin(train_sample['item_description'])]
print(f"Test size after removing overlaps: {len(test_clean)}")
test_clean.to_csv('test_sample20k_clean.csv', index=False)

(test_clean['category_name'].value_counts()).to_csv('test_clean_category_distribution.csv')
