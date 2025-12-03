import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { ModelPerformanceTable } from "@/components/model-table/model-performance-table";

export default function Results() {
	return (
		<section className="py-16 px-4">
				<div className="container mx-auto max-w-6xl">
					<div className="text-center mb-12">
						<h2 className="text-3xl font-bold text-gray-900 mb-4">
							Performance Results
						</h2>
						<p className="text-gray-600">
							Comprehensive evaluation across multiple datasets
							and metrics
						</p>
					</div>

					<Card className="overflow-hidden">
						<CardHeader>
							<CardTitle className="text-xl">
								Model Performance Comparison
							</CardTitle>
							<CardDescription>
								Accuracy, Precision, Recall, and F1-Score across
								different datasets and models
							</CardDescription>
						</CardHeader>
						<CardContent>
							<div className="overflow-x-auto">
								<ModelPerformanceTable />
							</div>

							<div className="mt-6 p-4 bg-blue-50 rounded-lg">
								<h4 className="font-semibold text-blue-900 mb-2">
									Key Findings
								</h4>
								<ul className="text-blue-800 space-y-1 text-sm">
									<li>
										• Multi-Att achieved the highest overall
										performance on Celeb-DF (Accuracy:
										97.92%, F1: 97.95%)
									</li>
									<li>
										• Frequency-based models (FreqNet,
										HiFi-FD) showed strong robustness to
										compression and noise
									</li>
									<li>
										• Attention-based models (Multi-Att,
										RFM) excelled on high-quality and
										challenging datasets
									</li>
									<li>
										• Cross-dataset generalization remains a
										challenge for all model categories
									</li>
								</ul>
							</div>
						</CardContent>
					</Card>
				</div>
			</section>
	)
}