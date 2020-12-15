# Capstone-3 Proposal

### Technologies of Interest
> I would like to get more experience with AWS, Spark, and Flask this capstone, so when reviewing the ideas, please keep that in mind and let me know if you think any project would be particularly good for those technologies.

## 1. Recipe Recommender

### Background: 
I've always loved food, and one of my hobbies is to try to recreate my favorite restaurant dishes at home. Since COVID began, my partner and I haven't been going to restaurants, and we've been cooking a ton. We have expanded (and exhausted) our recreated restaurant dishes portfolio. Now we are in the exploratory phase to add new recipes to our cook book.

### Minimum Viable Product:
Create a content-based recipe recommender (flask app) using features such as title and ingredients. For example, a user could search for a recipe title (or keywords) and/or specify ingredients they have on hand and the recommender will provide them a given number of recipe recommendations.

Getting to the data:
1. I came across this [website](https://recipe-search.typesense.org/) (open source recipe search app) 
2. I went to the [GitHub](https://github.com/typesense/showcase-recipe-search) and found a [link](https://github.com/Glorf/recipenlg) to the dataset on another GitHub repo
3. **The actual dataset is hosted on a project [website](https://recipenlg.cs.put.poznan.pl/)**
4. I agreed to the Terms and Conditions and downloaded the dataset. The dataset is too large to push to GitHub (even using git lfs, I think the limit is 2 GB), but I was able to load it in [Jupyter Notebook](https://github.com/coxem14/Capstone-3/blob/main/capstone_3.ipynb) just to get a look at it.
5. Notes from the GitHub: The dataset is 2.2 GB on disk, with ~2.2 million rows. It took 8 minutes to index this dataset on a 3-node Typesense cluster with 4vCPUs per node and the index was 2.7GB in RAM.
    > *I don't really know what this means, but it sounds pretty big, probably a little much for my macbook to handle*

### Possible Feature Engineering Opportunities
Things I wish this dataset included but doesn't immediately (the actual website may have this information):
* Other recipe features like prep time, cooking time, serving size, ethnicity/country of origin, and a rating from users of the website.
  > *I might be able to feature engineer some of this, but it would involve trying to manually label > 2 million recipes, webscrape > 2 million links, or doing something else that I can't immediately think of - let me know if you have ideas!*
* Indication of whether or not the recipe is for a breakfast, lunch, dinner, or dessert
  > *There are several titles that contain the words 'breakfast, lunch, dinner, or dessert' so I could definitely create categories for filtering by those keywords. For the recipes missing those keywords, I might be able to use clustering to categorize them.*
* Speaking of categories of food, types of food might be interesting to be able to filter by as well (such as 'soup, sandwich, salad, pizza, etc.')
  > *The user will be able to specify these in the keyword search if nothing else*

## 2. TBD

## 3. TBD