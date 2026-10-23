<div align="center">

<img src="https://playbees.firstcry.com/blog/wp-content/uploads/2025/12/Head-Shoulders-Knees-and-Toes-Lyrics-Nursery-Rhymes-For-Kids.jpg" width="760" alt="Head, Shoulders, Knees and Toes nursery-rhyme illustration">

<sub>Cover image selected by the owner. Source: <a href="https://www.bing.com/images/search?view=detailV2&ccid=R3yueXDs&id=615F980032BC4C1280484744A510D964D1306DA6">Bing Images / FirstCry PlayBees</a>.</sub>

</div>


# Head and shoulders, knees and toes

A reproducible starting point for the RSNA Knee Abnormality Detection competition.
This notebook turns each study's MRI series into scores for twelve findings, then
combines models by their ordering of studies.

This release faithfully reproduces **Mattia Angeli's public version 27** of
[Bend the Knee to the Dinosaurs](https://www.kaggle.com/code/mattiaangeli/bend-the-knee-to-the-dinosaurs?scriptVersionId=348195103).
That exact source version displays **0.940** on Kaggle. This is the source's result;
our version's score is established by its own completed competition submission.
Our previous verified checkpoints were Bee V16 at 0.938 and H&S V49 at 0.936.

The contribution here is an audited reproduction, original walkthrough, stage
artifacts, and execution checks. The model architecture and blend are inherited.
Every nonempty prediction cell retains the exact source text from version 27.

## Follow the prediction path

| Stage | What it contributes | Combination retained from the source |
|---|---|---|
| DINO | Twenty trained MRI members over ordered series and overlapping slice windows | Finding-specific window pooling and member ranks |
| A5 | Five DINOv3 folds with attention pooling over MRI slots | 45% of the DINO/A5 blend |
| RadImageNet | Reference heads and E13 heads applied to their recorded pixel layouts | Source fusion followed by its fixed calibrator |
| Raptor | Four CoAtNet views with different coverage and slice order | Probability weights 60%, 10%, 10%, 20% |
| Residual CoAtNet | Three released checkpoints, epochs 4, 6 and 8 | Equal mean of checkpoint ranks; 40% inside the CoAt branch |
| Final fusion | Transformer/Rad branch plus CoAt branch | CoAt weight 60%; lateral meniscus 100%; final percentile rank |

The intermediate Raptor/residual-CoAt blend is kept as a weighted rank value.
Ranking it again before the final fusion would change this recipe. Some inherited
internal stages use ordinal ranks; the final fusion uses average-tie percentile
ranks. These scores describe orderings and are not calibrated clinical probabilities.

## Run it

Use the attached input versions, the recorded Kaggle Python image, **GPU T4 ×2**,
and **Internet off**. Run every cell in order. No training occurs in this notebook.
Ordinary Kaggle runs contain three visible test studies; a competition submission
reruns the same code on hidden studies. Three visible rows can verify mechanics,
but cannot estimate the hidden-test AUC.

The last cell verifies every stage, checks study IDs and order, reconstructs the
final blend, and writes `reproduction_receipt.json` beside `submission.csv`.
The saved intermediate CSVs make each stage inspectable without rerunning models.


## 1. DINO: read the study and combine twenty members

The source reader uses DICOM geometry and MRI series descriptions to build its saved slot layout. Each trained member receives the same preprocessing used by its released package. Overlapping slice windows supply the finding-specific pooling votes.


## 2. A5: add five attention-pooling folds

These folds use their own saved DINOv3 configuration. Their contribution is fixed at 45% of the DINO/A5 blend. The source restores its earlier globals afterwards because the next stage reuses that reader.


## 3. RadImageNet: reproduce the reference and E13 branch

This stage retains the exact source layouts, head identities, and fixed calibration coefficients. The alternative Rad output is diagnostic only. The reference branch is the one passed forward.


## 4. CoAtNet: combine Raptor views and residual checkpoints

Four Raptor views are averaged at 60/10/10/20 before ranking. Three residual checkpoints contribute 40% inside this branch. The final fusion gives it 60% for most findings and 100% for lateral meniscus. The prediction code below is unchanged from the scored source.


## Sources, licenses, and limits

The assembled inference code is reproduced from Mattia Angeli's V27 notebook under
**Apache 2.0**. Its code comments and embedded notices are retained. Original
explanations and verification cells were added for this release. The source
notebook file is authenticated by SHA-256
`6b03919bbca5fe61e37c5ea6546609a771cd7db6503121f3e2dacdd8c9b6b052`.

| Source | Role and attribution |
|---|---|
| [pilkwang's RSNA weights](https://www.kaggle.com/datasets/pilkwang/rsna-knee-weights) | Released DINO MRI ensemble; dataset CC0-1.0, original backbone terms retained |
| [Mattia Angeli's fold weights](https://www.kaggle.com/datasets/mattiaangeli/knee-mri-fold-weights) | Five A5 folds and their saved inference configuration; dataset CC0-1.0, original backbone terms retained |
| [Johnathan Wagner / DreadDevelopment's Max-Span](https://www.kaggle.com/datasets/dreaddevelopment/raptor-knee-maxspan), [Native384](https://www.kaggle.com/datasets/dreaddevelopment/raptor-knee-native384), and [Native384 Dense](https://www.kaggle.com/datasets/dreaddevelopment/raptor-knee-native384dense) | Raptor CoAtNet checkpoints and view contracts; all three datasets CC0-1.0 |
| [Mattia Angeli's residual CoAtNet release](https://www.kaggle.com/datasets/mattiaangeli/rsna-knee-coat-resgated-ep10-top3) | Epochs 4/6/8 and authenticated runtime package; dataset CC0-1.0 |
| [Antoine's reference Rad heads](https://www.kaggle.com/datasets/antoinegg1/rsna-knee-e9-radimagenet-heads-v15) | Operative reference-v15 heads; **CC-BY-NC-SA-4.0** for the generated model artifacts |
| [Marwan's RadImageNet mirror](https://www.kaggle.com/datasets/marwanmath/resnet-50-radimagenet-marwan) | BMEII-AI / RadImageNet ResNet-50 encoder; Kaggle weight metadata **CC-BY-NC-SA-4.0** |
| [Sofia Anjenje's E11](https://www.kaggle.com/code/sofiaanjenje/rsna-knee-e11-train) and [E13](https://www.kaggle.com/code/sofiaanjenje/rsna-knee-e13-train) | Public training implementations and E13 heads; notebook code Apache 2.0, with the inherited encoder's model terms retained |
| [Meta DINOv2](https://www.kaggle.com/models/metaresearch/dinov2/PyTorch/small/1) | Attached DINOv2 backbone; retain its model license |
| [Meta DINOv3](https://github.com/facebookresearch/dinov3) | A5 uses `vit_small_patch16_dinov3.lvd1689m`; the [DINOv3 License](https://github.com/facebookresearch/dinov3/blob/main/LICENSE.md) remains distinct from the fold dataset's CC0 grant |
| [RSNA competition](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection) | MRI studies and competition data; competition access and data terms apply |
| [OpenCV Python 4.12.0.88](https://pypi.org/project/opencv-python-headless/4.12.0.88/) | Exact wheel SHA-256 matches the official release; Apache 2.0, with bundled third-party notices retained |

Code licensing does not relicense pretrained weights. The RadImageNet components
carry noncommercial and share-alike conditions separately from Apache-licensed
notebook code. This is a Kaggle research checkpoint. Review the original model
and dataset terms before redistributing weights or considering another use.
License texts: [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0),
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/), and
[CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/).
Dependencies retained solely for exact source compatibility are listed in the
attached-input manifest; attaching a resource does not make it an active model.
Our historical V52 alternative Rad heads produce a diagnostic CSV; the final
writer uses reference-v15 and E13. They are not an extra final ensemble member.

The inherited checkpoints and calibrator used their authors' training and model
selection procedures, including official labels in some releases. We have not
created a new independent validation set by reproducing them. No new Gold-label
evaluation or leaderboard-driven weight search occurs here. A public score on
this competition does not establish performance on other patients or scanners.

## Verify the final artifact

The following checks inspect the predictions already computed. They do not alter
the ensemble. A missing branch, study-level model fallback, UID mismatch, or
nonfinite output fails the run before it can qualify for submission.


<details><summary>Full license: APACHE-2.0.txt</summary>


                                 Apache License
                           Version 2.0, January 2004
                        http://www.apache.org/licenses/

   TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION

   1. Definitions.

      "License" shall mean the terms and conditions for use, reproduction,
      and distribution as defined by Sections 1 through 9 of this document.

      "Licensor" shall mean the copyright owner or entity authorized by
      the copyright owner that is granting the License.

      "Legal Entity" shall mean the union of the acting entity and all
      other entities that control, are controlled by, or are under common
      control with that entity. For the purposes of this definition,
      "control" means (i) the power, direct or indirect, to cause the
      direction or management of such entity, whether by contract or
      otherwise, or (ii) ownership of fifty percent (50%) or more of the
      outstanding shares, or (iii) beneficial ownership of such entity.

      "You" (or "Your") shall mean an individual or Legal Entity
      exercising permissions granted by this License.

      "Source" form shall mean the preferred form for making modifications,
      including but not limited to software source code, documentation
      source, and configuration files.

      "Object" form shall mean any form resulting from mechanical
      transformation or translation of a Source form, including but
      not limited to compiled object code, generated documentation,
      and conversions to other media types.

      "Work" shall mean the work of authorship, whether in Source or
      Object form, made available under the License, as indicated by a
      copyright notice that is included in or attached to the work
      (an example is provided in the Appendix below).

      "Derivative Works" shall mean any work, whether in Source or Object
      form, that is based on (or derived from) the Work and for which the
      editorial revisions, annotations, elaborations, or other modifications
      represent, as a whole, an original work of authorship. For the purposes
      of this License, Derivative Works shall not include works that remain
      separable from, or merely link (or bind by name) to the interfaces of,
      the Work and Derivative Works thereof.

      "Contribution" shall mean any work of authorship, including
      the original version of the Work and any modifications or additions
      to that Work or Derivative Works thereof, that is intentionally
      submitted to Licensor for inclusion in the Work by the copyright owner
      or by an individual or Legal Entity authorized to submit on behalf of
      the copyright owner. For the purposes of this definition, "submitted"
      means any form of electronic, verbal, or written communication sent
      to the Licensor or its representatives, including but not limited to
      communication on electronic mailing lists, source code control systems,
      and issue tracking systems that are managed by, or on behalf of, the
      Licensor for the purpose of discussing and improving the Work, but
      excluding communication that is conspicuously marked or otherwise
      designated in writing by the copyright owner as "Not a Contribution."

      "Contributor" shall mean Licensor and any individual or Legal Entity
      on behalf of whom a Contribution has been received by Licensor and
      subsequently incorporated within the Work.

   2. Grant of Copyright License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      copyright license to reproduce, prepare Derivative Works of,
      publicly display, publicly perform, sublicense, and distribute the
      Work and such Derivative Works in Source or Object form.

   3. Grant of Patent License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      (except as stated in this section) patent license to make, have made,
      use, offer to sell, sell, import, and otherwise transfer the Work,
      where such license applies only to those patent claims licensable
      by such Contributor that are necessarily infringed by their
      Contribution(s) alone or by combination of their Contribution(s)
      with the Work to which such Contribution(s) was submitted. If You
      institute patent litigation against any entity (including a
      cross-claim or counterclaim in a lawsuit) alleging that the Work
      or a Contribution incorporated within the Work constitutes direct
      or contributory patent infringement, then any patent licenses
      granted to You under this License for that Work shall terminate
      as of the date such litigation is filed.

   4. Redistribution. You may reproduce and distribute copies of the
      Work or Derivative Works thereof in any medium, with or without
      modifications, and in Source or Object form, provided that You
      meet the following conditions:

      (a) You must give any other recipients of the Work or
          Derivative Works a copy of this License; and

      (b) You must cause any modified files to carry prominent notices
          stating that You changed the files; and

      (c) You must retain, in the Source form of any Derivative Works
          that You distribute, all copyright, patent, trademark, and
          attribution notices from the Source form of the Work,
          excluding those notices that do not pertain to any part of
          the Derivative Works; and

      (d) If the Work includes a "NOTICE" text file as part of its
          distribution, then any Derivative Works that You distribute must
          include a readable copy of the attribution notices contained
          within such NOTICE file, excluding those notices that do not
          pertain to any part of the Derivative Works, in at least one
          of the following places: within a NOTICE text file distributed
          as part of the Derivative Works; within the Source form or
          documentation, if provided along with the Derivative Works; or,
          within a display generated by the Derivative Works, if and
          wherever such third-party notices normally appear. The contents
          of the NOTICE file are for informational purposes only and
          do not modify the License. You may add Your own attribution
          notices within Derivative Works that You distribute, alongside
          or as an addendum to the NOTICE text from the Work, provided
          that such additional attribution notices cannot be construed
          as modifying the License.

      You may add Your own copyright statement to Your modifications and
      may provide additional or different license terms and conditions
      for use, reproduction, or distribution of Your modifications, or
      for any such Derivative Works as a whole, provided Your use,
      reproduction, and distribution of the Work otherwise complies with
      the conditions stated in this License.

   5. Submission of Contributions. Unless You explicitly state otherwise,
      any Contribution intentionally submitted for inclusion in the Work
      by You to the Licensor shall be under the terms and conditions of
      this License, without any additional terms or conditions.
      Notwithstanding the above, nothing herein shall supersede or modify
      the terms of any separate license agreement you may have executed
      with Licensor regarding such Contributions.

   6. Trademarks. This License does not grant permission to use the trade
      names, trademarks, service marks, or product names of the Licensor,
      except as required for reasonable and customary use in describing the
      origin of the Work and reproducing the content of the NOTICE file.

   7. Disclaimer of Warranty. Unless required by applicable law or
      agreed to in writing, Licensor provides the Work (and each
      Contributor provides its Contributions) on an "AS IS" BASIS,
      WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
      implied, including, without limitation, any warranties or conditions
      of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A
      PARTICULAR PURPOSE. You are solely responsible for determining the
      appropriateness of using or redistributing the Work and assume any
      risks associated with Your exercise of permissions under this License.

   8. Limitation of Liability. In no event and under no legal theory,
      whether in tort (including negligence), contract, or otherwise,
      unless required by applicable law (such as deliberate and grossly
      negligent acts) or agreed to in writing, shall any Contributor be
      liable to You for damages, including any direct, indirect, special,
      incidental, or consequential damages of any character arising as a
      result of this License or out of the use or inability to use the
      Work (including but not limited to damages for loss of goodwill,
      work stoppage, computer failure or malfunction, or any and all
      other commercial damages or losses), even if such Contributor
      has been advised of the possibility of such damages.

   9. Accepting Warranty or Additional Liability. While redistributing
      the Work or Derivative Works thereof, You may choose to offer,
      and charge a fee for, acceptance of support, warranty, indemnity,
      or other liability obligations and/or rights consistent with this
      License. However, in accepting such obligations, You may act only
      on Your own behalf and on Your sole responsibility, not on behalf
      of any other Contributor, and only if You agree to indemnify,
      defend, and hold each Contributor harmless for any liability
      incurred by, or claims asserted against, such Contributor by reason
      of your accepting any such warranty or additional liability.

   END OF TERMS AND CONDITIONS

   APPENDIX: How to apply the Apache License to your work.

      To apply the Apache License to your work, attach the following
      boilerplate notice, with the fields enclosed by brackets "[]"
      replaced with your own identifying information. (Don't include
      the brackets!)  The text should be enclosed in the appropriate
      comment syntax for the file format. We also recommend that a
      file or class name and description of purpose be included on the
      same "printed page" as the copyright notice for easier
      identification within third-party archives.

   Copyright [yyyy] [name of copyright owner]

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.


</details>


<details><summary>Full license: DINOv3-LICENSE.md</summary>

# DINOv3 License

*Last Updated: August 19, 2025*

**“Agreement”** means the terms and conditions for use, reproduction, distribution and modification of the DINO Materials set forth herein.

**“DINO Materials”** means, collectively, Documentation and the models, software and algorithms, including machine-learning model code, trained model weights, inference-enabling code, training-enabling code, fine-tuning enabling code, and other elements of the foregoing distributed by Meta and made available under this Agreement.

**“Documentation”** means the specifications, manuals and documentation accompanying
DINO Materials distributed by Meta.

**“Licensee”** or **“you”** means you, or your employer or any other person or entity (if you are entering into this Agreement on such person or entity’s behalf), of the age required under applicable laws, rules or regulations to provide legal consent and that has legal authority to bind your employer or such other person or entity if you are entering in this Agreement on their behalf.

**“Meta”** or **“we”** means Meta Platforms Ireland Limited (if you are located in or, if you are an entity, your principal place of business is in the EEA or Switzerland) or Meta Platforms, Inc. (if you are located outside of the EEA or Switzerland).

**“Sanctions”** means any economic or trade sanctions or restrictions administered or enforced by the United States (including the Office of Foreign Assets Control of the U.S. Department of the Treasury (“OFAC”), the U.S. Department of State and the U.S. Department of Commerce), the United Nations, the European Union, or the United Kingdom.

**“Trade Controls”** means any of the following: Sanctions and applicable export and import controls.

By clicking “I Accept” below or by using or distributing any portion or element of the DINO Materials, you agree to be bound by this Agreement.

## 1. License Rights and Redistribution.

a. <ins>Grant of Rights</ins>. You are granted a non-exclusive, worldwide, non-transferable and royalty-free limited license under Meta’s intellectual property or other rights owned by Meta embodied in the DINO Materials to use, reproduce, distribute, copy, create derivative works of, and make modifications to the DINO Materials.

b. <ins>Redistribution and Use</ins>.

i. Distribution of DINO Materials, and any derivative works thereof, are subject to the terms of this Agreement. If you distribute or make the DINO Materials, or any derivative works thereof, available to a third party, you may only do so under the terms of this Agreement and you shall provide a copy of this Agreement with any such DINO Materials.

ii.  If you submit for publication the results of research you perform on, using, or otherwise in connection with DINO Materials, you must acknowledge the use of DINO Materials in your publication.

iii. Your use of the DINO Materials must comply with applicable laws and regulations, including Trade Control Laws and applicable privacy and data protection laws.

iv. Your use of the DINO Materials will not involve or encourage others to reverse engineer, decompile or discover the underlying components of the DINO Materials.

v. You are not the target of Trade Controls and your use of DINO Materials must comply with Trade Controls. You agree not to use, or permit others to use, DINO Materials for any activities subject to the International Traffic in Arms Regulations (ITAR) or end uses prohibited by Trade Controls, including those related to military or warfare purposes, nuclear industries or applications, espionage, or the development or use of guns or illegal weapons.

## 2. User Support.

Your use of the DINO Materials is done at your own discretion; Meta does not process any information nor provide any service in relation to such use.  Meta is under no obligation to provide any support services for the DINO Materials. Any support provided is “as is”, “with all faults”, and without warranty of any kind.

## 3. Disclaimer of Warranty.

UNLESS REQUIRED BY APPLICABLE LAW, THE DINO MATERIALS AND ANY OUTPUT AND RESULTS THEREFROM ARE PROVIDED ON AN “AS IS” BASIS, WITHOUT WARRANTIES OF ANY KIND, AND META DISCLAIMS ALL WARRANTIES OF ANY KIND, BOTH EXPRESS AND IMPLIED, INCLUDING, WITHOUT LIMITATION, ANY WARRANTIES OF TITLE, NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. YOU ARE SOLELY RESPONSIBLE FOR DETERMINING THE APPROPRIATENESS OF USING OR REDISTRIBUTING THE DINO MATERIALS AND ASSUME ANY RISKS ASSOCIATED WITH YOUR USE OF THE DINO MATERIALS AND ANY OUTPUT AND RESULTS.

## 4. Limitation of Liability.

IN NO EVENT WILL META OR ITS AFFILIATES BE LIABLE UNDER ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, TORT, NEGLIGENCE, PRODUCTS LIABILITY, OR OTHERWISE, ARISING OUT OF THIS AGREEMENT, FOR ANY LOST PROFITS OR ANY DIRECT OR INDIRECT, SPECIAL, CONSEQUENTIAL, INCIDENTAL, EXEMPLARY OR PUNITIVE DAMAGES, EVEN IF META OR ITS AFFILIATES HAVE BEEN ADVISED OF THE POSSIBILITY OF ANY OF THE FOREGOING.

## 5. Intellectual Property.

a. Subject to Meta’s ownership of DINO Materials and derivatives made by or for Meta, with respect to any derivative works and modifications of the DINO Materials that are made by you, as between you and Meta, you are and will be the owner of such derivative works and modifications.

b. If you institute litigation or other proceedings against Meta or any entity (including a cross-claim or counterclaim in a lawsuit) alleging that the DINO Materials, outputs or results, or any portion of any of the foregoing, constitutes infringement of intellectual property or other rights owned or licensable by you, then any licenses granted to you under this Agreement shall terminate as of the date such litigation or claim is filed or instituted. You will indemnify and hold harmless Meta from and against any claim by any third party arising out of or related to your use or distribution of the DINO Materials.

## 6. Term and Termination.

The term of this Agreement will commence upon your acceptance of this Agreement or access to the DINO Materials and will continue in full force and effect until terminated in accordance with the terms and conditions herein. Meta may terminate this Agreement if you are in breach of any term or condition of this Agreement. Upon termination of this Agreement, you shall delete and cease use of the DINO Materials. Sections 3, 4 and 7 shall survive the termination of this Agreement.

## 7. Governing Law and Jurisdiction.

This Agreement will be governed and construed under the laws of the State of California without regard to choice of law principles, and the UN Convention on Contracts for the International Sale of Goods does not apply to this Agreement. The courts of California shall have exclusive jurisdiction of any dispute arising out of this Agreement.

## 8. Modifications and Amendments.

Meta may modify this Agreement from time to time; provided that they are similar in spirit to the current version of the Agreement, but may differ in detail to address new problems or concerns. All such changes will be effective immediately. Your continued use of the DINO Materials after any modification to this Agreement constitutes your agreement to such modification. Except as provided in this Agreement, no modification or addition to any provision of this Agreement will be binding unless it is in writing and signed by an authorized representative of both you and Meta.


</details>


### Pinned attached inputs

- `dreaddevelopment/raptor-knee-maxspan/1`
- `dreaddevelopment/raptor-knee-native384/1`
- `dreaddevelopment/raptor-knee-native384dense/1`
- `mattiaangeli/knee-mri-fold-weights/2`
- `mattiaangeli/opencv-python-headless-4120088-x86/1`
- `marwanmath/resnet-50-radimagenet-marwan/1`
- `mattiaangeli/rsna-knee-coat-resgated-ep10-top3/3`
- `antoinegg1/rsna-knee-e11-diverse-heads-v20/1`
- `antoinegg1/rsna-knee-e9-radimagenet-heads-v15/3`
- `pilkwang/rsna-knee-llm-labels/1`
- `prvsiyan/rsna-knee-v52-radimagenet-heads-20260812/1`
- `pilkwang/rsna-knee-weights/1`
- `sofiaanjenje/rsna-knee-e11-train/2`
- `sofiaanjenje/rsna-knee-e13-train/1`
- `metaresearch/dinov2/PyTorch/small/1`
