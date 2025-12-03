import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { ModelPerformanceTable } from "@/components/model-table/model-performance-table";
import Image from "next/image";
export default function Results() {
	return (
		// <section className="py-16 px-4">
		// 		<div className="container mx-auto max-w-6xl">
		// 			<div className="text-center mb-12">
		// 				<h2 className="text-3xl font-bold text-gray-900 mb-4">
		// 					Performance Results
		// 				</h2>
		// 				<p className="text-gray-600">
		// 					Comprehensive evaluation across multiple datasets
		// 					and metrics
		// 				</p>
		// 			</div>

		// 			<Card className="overflow-hidden">
		// 				<CardHeader>
		// 					<CardTitle className="text-xl">
		// 						Model Performance Comparison
		// 					</CardTitle>
		// 					<CardDescription>
		// 						Accuracy, Precision, Recall, and F1-Score across
		// 						different datasets and models
		// 					</CardDescription>
		// 				</CardHeader>
		// 				<CardContent>
		// 					<div className="overflow-x-auto">
		// 						<ModelPerformanceTable />
		// 					</div>

		// 					<div className="mt-6 p-4 bg-blue-50 rounded-lg">
		// 						<h4 className="font-semibold text-blue-900 mb-2">
		// 							Key Findings
		// 						</h4>
		// 						<ul className="text-blue-800 space-y-1 text-sm">
		// 							<li>
		// 								• Multi-Att achieved the highest overall
		// 								performance on Celeb-DF (Accuracy:
		// 								97.92%, F1: 97.95%)
		// 							</li>
		// 							<li>
		// 								• Frequency-based models (FreqNet,
		// 								HiFi-FD) showed strong robustness to
		// 								compression and noise
		// 							</li>
		// 							<li>
		// 								• Attention-based models (Multi-Att,
		// 								RFM) excelled on high-quality and
		// 								challenging datasets
		// 							</li>
		// 							<li>
		// 								• Cross-dataset generalization remains a
		// 								challenge for all model categories
		// 							</li>
		// 						</ul>
		// 					</div>
		// 				</CardContent>
		// 			</Card>
		// 		</div>
		// 	</section>

        <section className="py-16 px-20 bg-white">

            <div className="container mb-8"> 
            <div className="container mx-auto max-w-5xl mb-8">
                
                        <div className="text-center mb-4">
                            <h2 className="text-3xl font-bold text-gray-900">
                                Our Results
                            </h2>
                        </div>
                <div className="text-1xl font-bold text-gray-900">Baseline & Altered Evaluation</div>
                        <p>Spatial models such as Xception and RECCE showed stable, reliable performance across datasets, while frequency-based models like FreqNet and HiFi-FD excelled at detecting high-frequency artifacts but struggled with compressed or low-quality inputs. Attention-based models, including Multi-Att and RFM, demonstrated strong localization of manipulated regions, with ablation results confirming that guided attention significantly improves detection accuracy.</p>
                        <Image 
                            src="/heatmaps.png" 
                            alt="Results" 
                            width={1920} 
                            height={1080} 
                            className="w-full h-auto object-contain" 
                        />
                        <p  className="text-gray-700 text-sm">Figure: Heatmaps of spatial (Xception, RECCE), frequency-based (FreqNet, HiFi-FD), and attention-based (Multi-Att, RFM) face forgery detection models evaluated under a wide range of perturbations. Each heatmap illustrates the models' accuracy, precision, recall, and F1-score.</p>

                        <Image 
                            src="/bargraphs.png" 
                            alt="Results" 
                            width={1920} 
                            height={1080} 
                            className="w-full h-auto object-contain mt-8" 
                        />
                        <p  className="text-gray-700 text-sm">Figure: Bar graphs showing average accuracy, precision, recall, and F1-score of spatial (Xception, RECCE), frequency-based (FreqNet, HiFi-FD), and attention-based (Multi-Att, RFM) face forgery detection models evaluated under a wide range of perturbations.</p>
                        
                        <div className="container mx-auto max-w-5xl mb-8 mt-8">
                            <div className="text-1xl font-bold text-gray-900">Cross-dataset Evaluation</div>
                            <p>Cross-dataset evaluations show that attention-based models, especially Multi-Att, generalize best to unfamiliar manipulation styles, while spatial models remain relatively stable and frequency-based models generalize poorly. More challenging datasets like WildDeepFake and FFc40 expose these weaknesses, reinforcing that flexible spatial reasoning and training strategies that address distribution shifts are essential for reliable real-world deepfake detection.</p>
                            <div className="flex flex-col items-center justify-center w-full">
                                <Image 
                                    src="/cross-dataset.png" 
                                    alt="Results" 
                                    width={1600} 
                                    height={900} 
                                    className="max-w-5xl w-full h-auto object-contain mx-auto mt-4" 
                                />
                                <p  className="text-gray-700 text-sm">Figure: Cross-dataset evaluations show that attention-based models, especially Multi-Att, generalize best to unfamiliar manipulation styles, while spatial models remain relatively stable and frequency-based models generalize poorly. More challenging datasets like WildDeepFake and FFc40 expose these weaknesses, reinforcing that flexible spatial reasoning and training strategies that address distribution shifts are essential for reliable real-world deepfake detection.</p>
                            </div>
                        </div>
                    </div>
                    <div className="container mx-auto max-w-5xl mb-8">
                        <div className="text-1xl font-bold text-gray-900">Speed & Memory</div>
                        <div>Efficiency analysis shows that lightweight models like FreqNet and Multi-Att are best suited for real-time or resource-constrained environments, while more computationally intensive models such as RECCE and HiFi-FD excel in accuracy-critical applications. RFM and Xception provide a balanced compromise, emphasizing that practical deployment requires optimizing speed, memory use, and robustness to varied manipulation conditions.</div>
                        <div className="flex flex-col items-center justify-center w-full">
                            <Image 
                                src="/speed.png" 
                                alt="Results" 
                                width={1600} 
                                height={900} 
                                className="max-w-5xl w-full h-auto object-contain mx-auto mt-4" 
                            />
                            <p  className="text-gray-700 text-sm">Figure: Speed and memory usage of spatial (Xception, RECCE), frequency-based (FreqNet, HiFi-FD), and attention-based (Multi-Att, RFM) face forgery detection models evaluated.</p>
                        </div>
                    </div>


            </div>
        </section>
	)
}