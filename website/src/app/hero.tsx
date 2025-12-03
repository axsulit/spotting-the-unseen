import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { FileText, Database } from "lucide-react";

export default function Hero() {
	return (
		<section className="py-16 px-4">
				<div className="container mx-auto max-w-4xl text-center">
					<div className="mb-4 overflow-x-auto w-full hidden sm:block">
						<Badge
							variant="secondary"
							className="whitespace-nowrap px-4 py-2 min-w-max"
						>
							Face Forgery Detection • Comparative Model Analysis
							• Dataset Benchmarking
						</Badge>
					</div>
					<h1 className="text-4xl md:text-5xl font-bold text-gray-900 mb-6 leading-tight">
						Spotting the Unseen: A Comprehensive Analysis of Face
						Forgery Detection Models
					</h1>
					<div className="mb-6 flex flex-wrap justify-center gap-2 text-gray-700 text-base font-medium">
						<span>Isiah Reuben C. Exconde</span>
						<span>•</span>
						<span>Zhoe Aeris F. Gon Gon</span>
						<span>•</span>
						<span>Anne Gabrielle M. Sulit</span>
						<span>•</span>
						<span>Ysobella D. Torio</span>
					</div>
					<p className="text-xl text-gray-600 mb-8 max-w-3xl mx-auto leading-relaxed">
						Investigating the robustness of state-of-the-art
						deepfake detection models under real-world conditions
						and challenging scenarios.
					</p>
					<div className="flex flex-wrap justify-center gap-4">
						<Button
							size="lg"
							className="bg-blue-600 hover:bg-blue-700"
						>
							<FileText className="w-5 h-5 mr-2" />
							Read Paper
						</Button>
						<Button
							asChild
							variant="outline"
							size="lg"
							className="bg-white"
						>
							<a
								href="https://github.com/axsulit/spotting-the-unseen/tree/main/datasets"
								target="_blank"
								rel="noopener noreferrer"
							>
								<Database className="w-5 h-5 mr-2" />
								View Datasets
							</a>
						</Button>
					</div>
				</div>
			</section>
	)
}