# RecSys2022 by Dressipi
<p align="center">
  <a href="http://recsys.deib.polimi.it">
  <img width="100%" src="https://i.imgur.com/tm9mSuM.png" alt="Recommender System 2018 Challenge Polimi" />
  </a>
</p>

## About the Challenge
The [RecSys Challenge 2022](https://recsys.acm.org/recsys22/) was organized by [Dressipi](https://dressipi.com/), Bruce Ferwerda ([Jönköping University, Sweden](https://ju.se/en)), Saikishore Kalloori ([ETH Zürich, Switzerland](https://ethz.ch/en.html)), and Abhishek Srivastava ([IIM Visakhapatnam, India](https://www.iimv.ac.in/)).

The challenge focused on fashion recommendations.
Given user sessions, purchase data and content data about items, the task was to accurately predict which fashion item will be bought at the end of the session.

The Dataset is available at the following [link](https://www.dressipi-recsys2022.com/profile/download_dataset).

## Team Members

We participated in the challenge as Boston Team Party, a team of 7 MSc students from Politecnico di Milano:

* **[Nicola Della Volpe](https://github.com/NickdV99)**

* **[Lorenzo Mainetti](https://github.com/LorenzoMainetti)**

* **[Alessio Martignetti](https://github.com/alemartignetti)**

* **[Andrea Menta](https://github.com/Menta99)**

* **[Riccardo Pala](https://github.com/RikyPala)**

* **[Giacomo Polvanesi](https://github.com/polva98)**

* **[Francesco Sammarco](https://github.com/KingPowa)**

We worked under the supervision of:
* **[Fernando B. Peréz Maurera](https://github.com/fernandobperezm)**

* **[Cesare Bernardis](https://github.com/cesarebernardis)**

* **[Maurizio Ferrari Dacrema](https://github.com/maurizioFD)**

## Structure of the Repository

The repository is divided in the following parts:

* [Notebooks](https://github.com/KingPowa/Rec_Sys_2022_Boston_Team/tree/main/Notebooks), selection of notebooks used to explore the dataset and generate custom attributes

* [RecSys_Course_AT_PoliMi](https://github.com/KingPowa/Rec_Sys_2022_Boston_Team/tree/main/RecSys_Course_AT_PoliMi), fork of the course repository 
enriched with the GRU4Rec implementation by [Theano](https://github.com/hidasib/GRU4Rec), other custom models, utilities to handle the dataset, train models and perform inference

* [optimizer_files](https://github.com/KingPowa/Rec_Sys_2022_Boston_Team/tree/main/optimizer_files), scripts based on Opuna or Bayesian Optimization used to perform hyperparameter tuning of the models involved in the candidate generation

* [booster](https://github.com/KingPowa/Rec_Sys_2022_Boston_Team/tree/main/booster), scripts related to LightGBM, involving the creation of its dataset, the hyperparameter tuning and the inference

## Paper

It's available a [paper](https://github.com/KingPowa/Rec_Sys_2022_Boston_Team/tree/main/) based on our experience in the challenge, describing our discoveries and implementation choices.

## Results

Our model achieved a score of **0.1880** and **0.1845** in the public and private leaderboard respectively, granting us the **29th** place after the first round of evaluation
