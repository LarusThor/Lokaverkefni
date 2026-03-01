import pandas as pd 
from sklearn.model_selection import train_test_split

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
    "Electronics/Cell Phones & Accessories/Cases, Covers & Skins",
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

# Read and filter train.csv only
data = pd.read_csv("train.csv", sep='\t')

filtered = data[
    (data['item_description'] != 'No description yet') &
    (data['category_name'].notna()) &
    (data['price'].notna())  # ensure all rows have labels
].copy()
filtered = filtered[filtered['category_name'].isin(categories)]

# Build balanced pool from categories
category_samples = []
per_category = 130000 // len(categories)  # larger pool to give room for 3 way split
for category in categories:
    cat_data = filtered[filtered['category_name'] == category]
    cat_sample = cat_data.sample(min(len(cat_data), per_category), random_state=42)
    category_samples.append(cat_sample)

pool = pd.concat(category_samples, ignore_index=True)
print(f"Total pool size: {len(pool)}")

# Split into 70/15/15
train_set, temp = train_test_split(pool, test_size=0.30, random_state=42)
val_set, test_set = train_test_split(temp, test_size=0.50, random_state=42)

print(f"Train size: {len(train_set)}")
print(f"Validation size: {len(val_set)}")
print(f"Test size: {len(test_set)}")

# Verify no overlap between splits
assert len(set(train_set.index) & set(val_set.index)) == 0
assert len(set(train_set.index) & set(test_set.index)) == 0
assert len(set(val_set.index) & set(test_set.index)) == 0
print("No overlaps confirmed")

train_set.to_csv('train_sample.csv', index=False)
val_set.to_csv('validation_sample.csv', index=False)
test_set.to_csv('test_sample.csv', index=False)

# Category distribution check
train_set['category_name'].value_counts().to_csv('train_category_distribution.csv')
val_set['category_name'].value_counts().to_csv('validation_category_distribution.csv')
test_set['category_name'].value_counts().to_csv('test_category_distribution.csv')