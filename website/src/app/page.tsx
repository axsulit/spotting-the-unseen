'use client';
import { Header, Footer} from './header';
import Hero from './hero';
import Problem from './problem';
import Method from './method';
import Results from './results';
import Citation from './citation';
export default function AcademicProject() {
	return (
		<div className="min-h-screen bg-gradient-to-br from-slate-50 to-white">
            <Header />
			<Hero />

			<Problem />
			<Method />

			<Results />

            <Citation />

            <Footer />
        </div>
    );
}
