import { Badge } from "@/components/ui/badge";

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
					<div className="flex flex-wrap justify-center gap-2 text-base font-medium">
						<a href="https://sites.google.com/view/ice-exconde-e-portfolio/home" target="_blank" rel="noopener noreferrer" className="text-blue-600 hover:text-blue-700 hover:underline">Isiah Reuben C. Exconde</a>
						<span className="text-gray-700">•</span>
						<a href="https://zhoe-aeris.vercel.app/" target="_blank" rel="noopener noreferrer" className="text-blue-600 hover:text-blue-700 hover:underline">Zhoe Aeris F. Gon Gon</a>
						<span className="text-gray-700">•</span>
						<a href="https://axsulit.vercel.app/" target="_blank" rel="noopener noreferrer" className="text-blue-600 hover:text-blue-700 hover:underline">Anne Gabrielle M. Sulit</a>
						<span className="text-gray-700">•</span>
						<a href="https://bellatorio.com" target="_blank" rel="noopener noreferrer" className="text-blue-600 hover:text-blue-700 hover:underline">Ysobella D. Torio</a>
					</div>
				</div>
			</section>
	)
}