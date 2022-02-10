
## LT2318 Project Proposal

#### Title of the project: 
Image tagger with object-specific adjectives

#### Members of your group: 
Calvin Kullvén Liao

####  Link to Github repository: 
[https://github.com/chickenbror/cog-ai-project](https://github.com/chickenbror/cog-ai-project) (*Empty at the moment)

#### What is the problem that you are solving? Give also some key references, for example papers that your project is based on.

Image tagging is now prevalent on smartphones and cloud services. Most such applications can detect the objects in an image and generate tags for the objects, for example, "cat" or "tree", but they do not further describe the objects.

For the course project, I propose building a model for “image tagging with object-specific adjectives” which can generate tags containing modifier+noun such as “a yellow hairy dog”, “a pink flower”, etc. The model will be built upon readily available pre-trained models like ResNet for recognising objects in images. The task itself of generating tags for respective object in an image can be seen as a multi-class classification where each tag is a class.

#### What existing resources (datasets, code, etc) are there?
- Datasets of image-and-tags or image-and-caption tuples, readily available resource include Flickr, COCO, etc.

- Pretrained models: ResNet (one of its variants), MobileNet, etc.

- Frameworks and libraries: PyTorch, OpenCV, Scikit-learn, Matplotlib, and possibly more.

- The starting point would be finding suitable datasets of image-and-tags or image-and-captions pairs (although the former would probably be better), and the tags/captions should ideally include adjectives describing the objects, otherwise a separate way of teaching the model to recognize quality attributes must be designed; for example a sub-model or a layer to focus on recognizing the color and texture of an object's surface.


#### What tasks do you need to do to get your answers? Here think about work packages such as data preprocessing, coding, experiments, evaluation, reporting and presentation, etc. If you are working in a group make sure that they are equally distributed over all members of your team. While it is good and expected that each member contributes to all stages of the project it is also a good that for each work package you agree on the member who will primarily responsible for it.


- For data pre-processing, it is necessary to pre-process and normalize the images to turn them into tensors of the same shape and with values within the same range. Depending on the text description of the images, they will also need to be pre-processed / parsed to mark the nouns and adjectives associated with the contents in a given image.
- During development, validate a small subset of the dataset to test the accuracy
- Evaluate by comparing the generated tags and the targets


#### What is the expected result?
 - At least objects should be correctly detected and classified 
 - A detected object should be modified by one or more adjectives
   appropriate for the object's attributes and constraints. 
 - If no appropriate adjective can be found to describe an object, the model should learn to leave adjective blank (ie, only have the noun in the tag)

#### How will the project be evaluated?

 - Compare to gold target tags 
 - Compare to target adjectives (if
   available)
- Making a confusion matrix to see which objects are likely
   to get identified/misidentified

#### What challenges do you expect?
Possible challenges include: describing sizes (a small tree: relatively small or absolutely small?), constrains on the modifier-and-noun combination (a hairy cat (OK), but *a hairy window (bad/unnatural) ), multi-sense tags (mouse: rodent or computer apparatus?).

#### (An alternative proposal:)
Alternatively, I propose to work on ***Chinese GRE to predict Chinese spatial terms*** (cf. prepositions in English or Swedish, but more akin to circumfixing) from objects in images, inspired by Dallandrea et al (2015) that we discussed in Seminar 4, as well as Michell et al (2012) and Lu et al (2017) discussed in Seminar 3 during this course. The challenge would be finding appropriate dataset, and tools for tokenizing and parsing Chinese text as it is not space-segmented like English or Swedish.


#### References:

###### A. Ramisa, J. Wang, Y. Lu, E. Dellandrea, F. Moreno-Noguer, and R.    Gaizauskas. Combining geometric, textual and visual features for    predicting prepositions in image descriptions.  Download Combining    geometric, textual and visual features for predicting prepositions in    image descriptions. In Proceedings of the 2015 Conference on    Empirical Methods in Natural Language Processing, pages 214–220,    Lisbon, Portugal, 7–21 September 2015. Association for Computational    Linguistics.

 ###### Lu, C. Xiong, D. Parikh, and R. Socher. Knowing when to look: Adaptive attention via a visual sentinel for image captioning.    Arxiv:1612.01887 [cs CV], 6 June 2017.
       
###### Mitchell, X. Han, J. Dodge, A. Mensch, A. Goyal, A. Berg, K.    Yamaguchi, T. Berg, K. Stratos, and H. Daumé III. Midge: Generating    image descriptions from computer vision detections. Download Midge:    Generating image descriptions from computer vision detections. In    Proceedings of the 13th Conference of the European Chapter of the    Association for Computational Linguistics, pages 747–756. Association    for Computational Linguistics, 2012.

